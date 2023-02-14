import argparse
import logging
import math
import os
from collections import OrderedDict
import sys
import time
import datetime
from typing import List

import numpy as np
import torch

from detectron2 import model_zoo
import detectron2.utils.comm as comm
from detectron2.utils.logger import log_every_n_seconds
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    DatasetCatalog,
    MetadataCatalog,
)
import detectron2.data.transforms as T
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    hooks,
    launch,
)

from detectron2.engine.hooks import EvalHook, BestCheckpointer

from detectron2.evaluation import (
    COCOEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA

import coco_loaders
import custom_datasets
from data_utils import (
    generate_missed_detections_data,
    random_split_mixed_set,
    read_split_file,
    register_dataset,
)


def custom_args(epilog=None):
    parser = default_argument_parser(epilog=epilog)
    parser.add_argument(
        "--arch", type=str, choices=["mask_rcnn", "cascade_mask_rcnn"], required=True
    )
    parser.add_argument("--dataset", "-d", type=str, help="dataset to use")
    parser.add_argument(
        "--epochs", type=int, help="num epochs to train for", default=50
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
    parser.add_argument("--score_thresh", type=float)
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


def build_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")

    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type == "coco":
        return COCOEvaluator(
            dataset_name=dataset_name,
            output_dir=output_folder,
            use_fast_impl=False,
            allow_cached_coco=False,
        )
    else:
        raise NotImplementedError(
            f"No Evaluator for {dataset_name} with the type {evaluator_type}"
        )


class Trainer(DefaultTrainer):
    """ """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return build_evaluator(cfg, dataset_name, output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        dataloader_type = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).dataloader_type
        if dataloader_type == "coco":
            return coco_loaders.build_coco_train_loader(cfg=cfg)
        else:
            raise NotImplementedError("unsupported dataloader type")

    @classmethod
    def build_validation_loader(cls, cfg):
        dataloader_style = MetadataCatalog.get(cfg.DATASETS.VAL[0]).dataloader_type
        if dataloader_style == "coco":
            return coco_loaders.build_coco_val_loader(cfg=cfg)
        else:
            raise NotImplementedError("unsupported dataloader type")

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        dataloader_style = MetadataCatalog.get(cfg.DATASETS.TEST[0]).dataloader_type
        if dataloader_style == "coco":
            return coco_loaders.build_coco_test_loader(cfg=cfg)
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
                    eval_after_train=False,
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


def register_dataset_splits(
    datasets: List[List[str]],
    classes: List[str],
    dataloader_type: str,
    evaluator_type: str,
):
    for category, paths in zip(["train", "val", "test"], datasets):
        DatasetCatalog.register(
            name=f"mixed_{category}",
            func=lambda im_paths=paths: register_dataset(
                im_paths=im_paths, skip_no_pairs=True
            ),
        )
        MetadataCatalog.get(f"mixed_{category}").set(
            thing_classes=classes,
            dataloader_type=dataloader_type,
            evaluator_type=evaluator_type,
        )


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

    # LOAD DATASET
    if args.dataset == "mixed":
        random_split_mixed_set(
            img_dir=custom_datasets.MIXED_SRC,
            split_ratio=custom_datasets.MIXED_SPLIT_RATIO,
            seed=args.seed,
        )
        datasets = read_split_file(fpath=custom_datasets.MIXED_SPLIT_FILE_PATH)
        register_dataset_splits(
            datasets=datasets,
            classes=list(custom_datasets.CLASSES.keys()),
            dataloader_type="coco",
            evaluator_type="coco",
        )
    elif args.dataset == "md_mixed":
        if not os.path.exists(custom_datasets.MIXED_SPLIT_FILE_PATH):
            random_split_mixed_set(
                img_dir=custom_datasets.MIXED_SRC,
                split_ratio=custom_datasets.MIXED_SPLIT_RATIO,
                seed=args.seed,
            )
        if not os.path.exists(custom_datasets.MD_MIXED_SRC):
            log_every_n_seconds(
                logging.INFO, f"generating missed detections mixed dataset..."
            )
            generate_missed_detections_data(
                dataset_name="mixed", skip_no_pairs=args.skip_no_pairs
            )

        datasets = read_split_file(custom_datasets.MD_MIXED_SPLIT_FILE_PATH)
        register_dataset_splits(
            datasets=datasets,
            classes=list(custom_datasets.CLASSES.keys()),
            dataloader_type="default",
            evaluator_type="coco",
        )
    else:
        raise NotImplementedError("Unsupported Dataset")

    iter_per_epoch = iters_per_epoch(
        trainset_len=len(datasets[0]), batch_size=args.batch_size
    )

    # start from pretrained weights
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_cfg)

    # LOAD DATASETS and adjust ROI HEADS to match num classes
    cfg.DATASETS.TRAIN = (f"{args.dataset}_train",)
    cfg.DATASETS.VAL = (f"{args.dataset}_val",)
    cfg.DATASETS.TEST = (f"{args.dataset}_test",)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(custom_datasets.CLASSES.keys())

    # TRAINING LOGIC
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

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


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
