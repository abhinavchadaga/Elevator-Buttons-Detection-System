import logging
import math
import os
import time
import datetime

import numpy as np
import torch

from detectron2 import model_zoo
import detectron2.utils.comm as comm
from detectron2.utils.logger import log_every_n_seconds
from detectron2.config import get_cfg
from detectron2.data import (
    DatasetCatalog,
    MetadataCatalog,
    DatasetMapper,
    get_detection_dataset_dicts,
    build_detection_train_loader,
    build_detection_test_loader,
)
import detectron2.data.transforms as T
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
from detectron2.engine.hooks import EvalHook, BestCheckpointer

import coco_loaders
import elevator_datasets
from data_utils import (
    generate_missed_detections_data,
    random_split_mixed_set,
    read_split_file,
    register_dataset,
)


def custom_args(epilog=None):
    parser = default_argument_parser(epilog=epilog)
    parser.add_argument("--dataset", "-d", type=str, help="dataset to use")
    parser.add_argument(
        "--arch", type=str, choices=["mask_rcnn", "cascade_mask_rcnn"], required=True
    )
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


def validation_loop(model: torch.nn, dataloader: torch.utils.data.DataLoader) -> dict:
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


class Trainer(DefaultTrainer):
    """ """

    @classmethod
    def build_train_loader(cls, cfg):
        dataloader_type = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).dataloader_type
        if dataloader_type == "default":
            return build_detection_train_loader(
                cfg,
                mapper=DatasetMapper(cfg, is_train=True),
            )
        elif dataloader_type == "coco":
            return coco_loaders.build_coco_train_loader(cfg=cfg)
        else:
            raise NotImplementedError("unsupported dataloader type")

    @classmethod
    def build_validation_loader(cls, cfg):
        dataset_name = cfg.DATASETS.VAL[0]
        dataloader_style = MetadataCatalog.get(dataset_name).dataloader_type
        if dataloader_style == "default":
            return build_detection_test_loader(
                dataset=get_detection_dataset_dicts(
                    names=dataset_name, filter_empty=False
                ),
                mapper=DatasetMapper(cfg, is_train=True),
            )
        elif dataloader_style == "coco":
            return coco_loaders.build_coco_val_loader(cfg=cfg)
        else:
            raise NotImplementedError("unsupported dataloader type")

    def build_hooks(self) -> list:
        """
        Overwrite the evaluation loop and checkpoint using best validation loss

        Returns:
            list: list of hooks
        """
        hooks = super().build_hooks()
        # remove existing EvalHook
        if comm.is_main_process():
            del hooks[-2]
            del hooks[-2]
        else:
            del hooks[-1]
        hooks.extend(
            [
                EvalHook(
                    eval_period=self.cfg.TEST.EVAL_PERIOD,
                    eval_function=lambda: validation_loop(
                        self.model,
                        self.build_validation_loader(self.cfg),
                    ),
                    eval_after_train=True,
                ),
                BestCheckpointer(
                    eval_period=self.cfg.TEST.EVAL_PERIOD,
                    checkpointer=self.checkpointer,
                    val_metric="validation_loss",
                    mode="min",
                ),
            ]
        )
        return hooks


def iters_per_epoch(trainset_len: int, batch_size: int):
    return math.ceil(trainset_len // batch_size)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    model_cfg = (
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        if args.arch == "mask_rcnn"
        else "Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml"
    )
    cfg.merge_from_file(model_zoo.get_config_file(model_cfg))

    # split data and generate it if necessary
    if args.dataset == "mixed":
        random_split_mixed_set(
            img_dir=elevator_datasets.MIXED_SRC,
            split_ratio=elevator_datasets.MIXED_SPLIT_RATIO,
            seed=args.seed,
        )
        datasets = read_split_file(fpath=elevator_datasets.MIXED_SPLIT_FILE_PATH)
        dataloader_type = "coco"
    elif args.dataset == "md_mixed":
        if not os.path.exists(elevator_datasets.MIXED_SPLIT_FILE_PATH):
            random_split_mixed_set(
                img_dir=elevator_datasets.MIXED_SRC,
                split_ratio=elevator_datasets.MIXED_SPLIT_RATIO,
                seed=args.seed,
            )
        if not os.path.exists(elevator_datasets.MD_MIXED_SRC):
            log_every_n_seconds(
                logging.INFO, f"generating missed detections mixed dataset..."
            )
            generate_missed_detections_data(
                dataset_name="mixed", skip_no_pairs=args.skip_no_pairs
            )
        datasets = read_split_file(fpath=elevator_datasets.MD_MIXED_SPLIT_FILE_PATH)
        dataloader_type = "default"
    else:
        raise NotImplementedError("Unsupported Dataset")

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
            dataloader_type=dataloader_type,
            evaluator_type="coco",
        )

    iter_per_epoch = iters_per_epoch(
        trainset_len=len(datasets[0]), batch_size=args.batch_size
    )

    # start from pretrained weights or fine tuned weights if supplied
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_cfg)

    # assign datasets and adjust ROI HEADS to match num classes
    cfg.DATASETS.TRAIN = (f"{args.dataset}_train",)
    cfg.DATASETS.VAL = (f"{args.dataset}_val",)
    cfg.DATASETS.TEST = (f"{args.dataset}_test",)
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False
    cfg.INPUT.RANDOM_FLIP = "none"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(elevator_datasets.CLASSES.keys())

    # Calculate training logic
    cfg.SOLVER.MAX_ITER = iter_per_epoch * args.epochs
    cfg.TEST.EVAL_PERIOD = iter_per_epoch * args.eval_period
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.SOLVER.BASE_LR = args.base_lr
    cfg.SOLVER.STEPS = (
        [int(0.8 * cfg.SOLVER.MAX_ITER), int(0.9 * cfg.SOLVER.MAX_ITER)]
        if args.decay_lr
        else []
    )
    cfg.SOLVER.WARMUP_ITERS = int(0.4 * cfg.SOLVER.MAX_ITER)

    # SAVE DIR
    cfg.OUTPUT_DIR = args.output_dir
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    # train model
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    trainer.train()


if __name__ == "__main__":
    args = custom_args().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
