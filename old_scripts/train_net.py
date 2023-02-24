import datetime
import logging
import math
import time
from detectron2 import model_zoo
from detectron2.data.build import get_detection_dataset_dicts
from termcolor import cprint

import torch
import numpy as np

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import (
    AMPTrainer,
    SimpleTrainer,
    default_argument_parser,
    default_setup,
    default_writers,
    hooks,
    launch,
)
from detectron2.engine.defaults import create_ddp_model
from detectron2.utils import comm
from detectron2.utils.logger import log_every_n_seconds


from data_utils import read_split_file, register_dataset, random_split_mixed_set
import elevator_datasets
import mixed_dataloaders

logger = logging.getLogger("detectron2")


def custom_args(epilog=None):
    parser = default_argument_parser(epilog=epilog)
    parser.add_argument("--dataset", "-d", type=str, help="dataset to use")
    parser.add_argument(
        "--epochs", type=int, help="num epochs to train for", default=100
    )
    parser.add_argument(
        "--batch_size", "-b", type=int, default=2, help="images per batch"
    )
    parser.add_argument("--base_lr", type=float, default=0.00025)
    parser.add_argument("--decay_lr", type=str, default=True)
    parser.add_argument(
        "--skip_no_pairs",
        type=bool,
        default=True,
        help="skip features that don't have a pair attribute",
    )
    parser.add_argument(
        "--eval_period", type=int, default=2, help="number of epochs to eval after"
    )
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument(
        "--seed", type=int, default=10, help="seed for shuffling dataset splits"
    )
    return parser


def do_val(model: torch.nn, dataloader: torch.utils.data.DataLoader) -> dict:
    """
    Validate model on the given dataloader. Put model in training mode to output loss
    dict but do not backpropogate gradients. Largely adapted from train_loop.py and
    inference_on_dataset

    Args:
        model (torch.nn): model to validate
        dataloader (torch.utils.data.DataLoader): validation dataloader set to training
            mode

    Returns:
        dict: {"validation_loss": val_loss}
    """
    model.train()
    num_batches = len(dataloader)
    num_warmup = min(5, num_batches - 1)
    start_time = time.perf_counter()
    total_compute_time = 0
    losses = []
    for idx, data in enumerate(dataloader):
        if idx == num_warmup:
            start_time = time.perf_counter()
            total_compute_time = 0
        start_compute_time = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total_compute_time += time.perf_counter() - start_compute_time
        iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
        seconds_per_img = total_compute_time / iters_after_start
        if idx >= num_warmup * 2 or seconds_per_img > 5:
            total_seconds_per_img = (
                time.perf_counter() - start_time
            ) / iters_after_start
            eta = datetime.timedelta(
                seconds=int(total_seconds_per_img) * (num_batches - idx - 1)
            )
            log_every_n_seconds(
                logging.INFO,
                "Loss on Validation  done {}/{}. {:.4f} s / img. ETA={}".format(
                    idx + 1, num_batches, seconds_per_img, str(eta)
                ),
                n=5,
            )

        batch_loss_dict = model(data)
        batch_loss_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in batch_loss_dict.items()
        }
        total_batch_loss = sum(loss for loss in batch_loss_dict.values())
        losses.append(total_batch_loss)
    val_loss = np.mean(losses)
    comm.synchronize()
    log_every_n_seconds(logging.INFO, f"VALIDATION_LOSS: {val_loss:.5f}")
    return {"validation_loss": val_loss}


def do_train(args, cfg):
    """
    Args:
        cfg: an object with the following attributes:
            model: instantiate to a module
            dataloader.{train,test}: instantiate to dataloaders
            dataloader.evaluator: instantiate to evaluator for test set
            optimizer: instantaite to an optimizer
            lr_multiplier: instantiate to a fvcore scheduler
            train: other misc config defined in `configs/common/train.py`, including:
                output_dir (str)
                init_checkpoint (str)
                amp.enabled (bool)
                max_iter (int)
                eval_period, log_period (int)
                device (str)
                checkpointer (dict)
                ddp (dict)
    """
    model = instantiate(cfg.model)
    logger = logging.getLogger("detectron2")
    logger.info("Model:\n{}".format(model))
    model.to(cfg.train.device)

    cfg.optimizer.params.model = model
    optim = instantiate(cfg.optimizer)

    train_loader = instantiate(cfg.dataloader.train)

    model = create_ddp_model(model, **cfg.train.ddp)
    trainer = (AMPTrainer if cfg.train.amp.enabled else SimpleTrainer)(
        model, train_loader, optim
    )
    checkpointer = DetectionCheckpointer(
        model,
        cfg.train.output_dir,
        trainer=trainer,
    )
    trainer.register_hooks(
        [
            hooks.IterationTimer(),
            hooks.LRScheduler(scheduler=instantiate(cfg.lr_multiplier)),
            hooks.PeriodicWriter(
                default_writers(cfg.train.output_dir, cfg.train.max_iter),
                period=cfg.train.log_period,
            )
            if comm.is_main_process()
            else None,
            hooks.EvalHook(
                eval_period=cfg.train.eval_period,
                eval_function=lambda: do_val(
                    model,
                    instantiate(cfg.dataloader.val),
                ),
                eval_after_train=True,
            ),
            hooks.BestCheckpointer(
                eval_period=cfg.train.eval_period,
                checkpointer=checkpointer,
                val_metric="validation_loss",
                mode="min",
            ),
        ]
    )

    checkpointer.resume_or_load(cfg.train.init_checkpoint, resume=args.resume)
    if args.resume and checkpointer.has_checkpoint():
        # The checkpoint stores the training iteration that just finished, thus we start
        # at the next iteration
        start_iter = trainer.iter + 1
    else:
        start_iter = 0
    trainer.train(start_iter, cfg.train.max_iter)


def iters_per_epoch(trainset_len: int, batch_size: int):
    return math.ceil(trainset_len / batch_size)


def setup(cfg, args):
    if args.dataset == "mixed":
        random_split_mixed_set(
            elevator_datasets.MIXED_SRC, elevator_datasets.MIXED_SPLIT_RATIO, args.seed
        )
        datasets = read_split_file(elevator_datasets.MIXED_SPLIT_FILE_PATH)

        # register datasets
        for category, paths in zip(["train", "val", "test"], datasets):
            DatasetCatalog.register(
                name=f"{args.dataset}_{category}",
                func=lambda im_paths=paths: register_dataset(
                    im_paths=im_paths, skip_no_pairs=args.skip_no_pairs
                ),
            )
            MetadataCatalog.get(f"{args.dataset}_{category}").set(
                thing_classes=list(elevator_datasets.CLASSES.keys()),
            )
        cprint(
            f"TRAINSET SIZE: {len(get_detection_dataset_dicts('mixed_train'))}",
            "red",
            attrs=["bold"],
        )
        # freeze first two blocks of backbone
        cfg.model.backbone.bottom_up.freeze_at = 2
        cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/new_baselines/mask_rcnn_R_101_FPN_400ep_LSJ/42073830/model_final_f96b26.pkl"
        # turn off Sync Batch Norm (does not work on one GPU)
        cfg.model.backbone.bottom_up.stem.norm = "BN"
        cfg.model.backbone.bottom_up.stages.norm = "BN"
        cfg.model.backbone.norm = "BN"
        # match num roi heads to num classes in dataset
        cfg.model.roi_heads.num_classes = 2
        # replace dataloader
        cfg.dataloader = mixed_dataloaders.dataloader
        # adjust training settings for dataset
        cfg.train.max_iter = (
            iters_per_epoch(len(datasets[0]), args.batch_size) * args.epochs
        )
        cfg.train.eval_period = (
            iters_per_epoch(len(datasets[0]), args.batch_size) * args.eval_period
        )
        cfg.train.output_dir = args.output_dir
        # optimizer settings
        cfg.optimizer.lr = args.base_lr

        # learning rate settings
        cfg.lr_multiplier.scheduler.num_updates = cfg.train.max_iter
        # cfg.lr_multiplier.scheduler.milestones = [
        #     int(0.8 * cfg.train.max_iter),
        #     int(0.9 * cfg.train.max_iter),
        # ]
        cfg.lr_multiplier.scheduler.values = [1.0]
        cfg.lr_multiplier.scheduler.milestones = []
        cfg.lr_multiplier.warmup_factor = 500 / cfg.train.max_iter
        cfg.lr_multiplier.warmup_length = 0.067
    else:
        raise NotImplementedError("Unsupported Dataset")

    print(LazyConfig.to_py(cfg.lr_multiplier))
    default_setup(cfg, args)


def main(args):
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    setup(cfg, args)

    if args.eval_only:
        model = instantiate(cfg.model)
        model.to(cfg.train.device)
        model = create_ddp_model(model)
        DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
        print(do_val(cfg, model))
    else:
        do_train(args, cfg)


if __name__ == "__main__":
    args = custom_args().parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
