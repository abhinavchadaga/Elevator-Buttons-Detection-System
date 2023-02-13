import argparse
from datetime import datetime
import logging
import math
import os
import time
from typing import Tuple
import numpy as np

import torch

from detectron2.utils.logger import setup_logger, log_first_n, log_every_n_seconds

setup_logger()

from detectron2.utils import comm
from detectron2 import model_zoo
from detectron2.data import (
    DatasetCatalog,
    MetadataCatalog,
    DatasetMapper,
    get_detection_dataset_dicts,
    build_detection_train_loader,
    build_detection_test_loader,
)
import detectron2.data.transforms as T
from detectron2.config import get_cfg, CfgNode
from detectron2.engine import (
    DefaultTrainer,
    EvalHook,
    BestCheckpointer,
    DefaultPredictor,
)
from detectron2.evaluation import COCOEvaluator, inference_on_dataset


from data_utils import random_split_mixed_set, get_mixed_set_dicts, CLASSES


def epochs_to_iters(
    dataset_name: str, epochs: int, batch_size: int, eval_every_n_epochs: 5
) -> Tuple[int, int]:
    dataset = get_detection_dataset_dicts(dataset_name)
    N = len(dataset)
    del dataset
    iters_per_epoch = math.ceil(N // batch_size)
    return iters_per_epoch * epochs, iters_per_epoch * eval_every_n_epochs


def validation_loop(model: torch.nn, dataloader: torch.utils.data.DataLoader) -> dict:
    # mostly from inference_on_dataset
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
    log_first_n(
        logging.INFO, f"VALIDATION_LOSS: {val_loss:.5f}", key=("caller", "message")
    )
    return {"validation_loss": val_loss}


class CustomTrainer(DefaultTrainer):
    """Update Default Trainer's dataloader"""

    image_size = 1024

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(
            dataset_name=dataset_name,
            tasks=["bbox"],
            output_dir=output_folder,
            use_fast_impl=False,
            allow_cached_coco=False,
        )

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(
            dataset=get_detection_dataset_dicts(cfg.DATASETS.TRAIN),
            mapper=DatasetMapper(
                is_train=True,
                augmentations=[
                    T.ResizeScale(
                        min_scale=0.1,
                        max_scale=2.0,
                        target_height=cls.image_size,
                        target_width=cls.image_size,
                    ),
                    T.FixedSizeCrop(
                        crop_size=(cls.image_size, cls.image_size), pad=False
                    ),
                ],
                image_format="BGR",
                use_instance_mask=True,
                recompute_boxes=True,
            ),
            total_batch_size=cfg.SOLVER.IMS_PER_BATCH,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
        )

    @classmethod
    def build_validation_loader(cls, cfg):
        return build_detection_test_loader(
            dataset=get_detection_dataset_dicts(cfg.DATASETS.TEST[0]),
            mapper=DatasetMapper(
                cfg,
                is_train=True,
                augmentations=[
                    T.ResizeShortestEdge(
                        short_edge_length=cls.image_size,
                        max_size=cls.image_size,
                    )
                ],
                image_format="BGR",
            ),
        )

    @classmethod
    def build_test_loader(cls, cfg, dataset_name=None):
        return build_detection_test_loader(
            dataset=get_detection_dataset_dicts(
                cfg.DATASETS.TEST[1], filter_empty=False
            ),
            mapper=DatasetMapper(
                is_train=False,
                augmentations=[
                    T.ResizeShortestEdge(
                        short_edge_length=cls.image_size, max_size=cls.image_size
                    )
                ],
                image_format="BGR",
            ),
        )

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.extend(
            [
                EvalHook(
                    eval_period=self.cfg.TEST.CUSTOM_EVAL_PERIOD,
                    eval_function=lambda: validation_loop(
                        self._trainer.model, self.build_validation_loader(self.cfg)
                    ),
                    eval_after_train=True,
                ),
                BestCheckpointer(
                    eval_period=self.cfg.TEST.CUSTOM_EVAL_PERIOD,
                    checkpointer=self.checkpointer,
                    val_metric="validation_loss",
                    mode="min",
                ),
            ]
        )
        return hooks


def setup_cfg(
    arch: str,
    dataset_name: str,
    batch_size: int,
    epochs: int,
    eval_every_n_epochs: int,
    base_lr: int,
    decay_lr=bool,
    output_dir=str,
) -> CfgNode:
    cmrcnn = "Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml"
    mrcnn = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    model = mrcnn if arch == "mrcnn" else cmrcnn
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model))
    cfg.INPUT.RANDOM_FLIP = "none"  # turn off default augmentation
    cfg.DATASETS.TRAIN = (f"{dataset_name}_train",)
    cfg.DATASETS.TEST = (f"{dataset_name}_val", f"{dataset_name}_test")
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)
    cfg.SOLVER.IMS_PER_BATCH = batch_size
    cfg.SOLVER.BASE_LR = base_lr
    cfg.SOLVER.MAX_ITER, cfg.TEST.CUSTOM_EVAL_PERIOD = epochs_to_iters(
        dataset_name=f"{dataset_name}_train",
        epochs=epochs,
        batch_size=batch_size,
        eval_every_n_epochs=eval_every_n_epochs,
    )
    steps = []
    if decay_lr:
        steps.append(int(0.9 * cfg.SOLVER.MAX_ITER))
        steps.append(int(0.95 * cfg.SOLVER.MAX_ITER))
    cfg.SOLVER.STEPS = steps
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    cfg.OUTPUT_DIR = output_dir
    return cfg


def main():
    parser = argparse.ArgumentParser(
        description="training script for Detectron2 models"
    )
    parser.add_argument("--dataset", "-d", type=str, default="mixed")
    parser.add_argument(
        "--arch", "-a", type=str, default="mrcnn", choices=["mrcnn", "c-mrcnn"]
    )
    parser.add_argument("--batch_size", "-b", type=int, default=2)
    parser.add_argument("--base_lr", type=float, default=0.00025)
    parser.add_argument("--decay_lr", type=str, default=False)
    parser.add_argument("--epochs", "-e", type=int, default=50)
    parser.add_argument("--skip_no_pairs", type=bool, default=False)
    parser.add_argument("--output_dir", "-o", type=str, default="models/output")
    parser.add_argument("--eval_every_n_epochs", type=int, default=5)
    args = parser.parse_args()

    dataset_name = args.dataset

    log_first_n(
        logging.INFO,
        f"using {'mask-rcnn' if args.arch == 'mrcnn' else 'cascade mask-rcnn'}"
        + " architecture",
    )

    if dataset_name == "mixed":
        # split and register mixed dataset
        img_paths = random_split_mixed_set("data/panels/mixed")
        log_first_n(logging.INFO, f"SKIP_NO_PAIRS: {args.skip_no_pairs}")
        for name, paths in zip(["train", "val", "test"], img_paths):
            DatasetCatalog.register(
                name=f"mixed_{name}",
                func=lambda im_paths=paths: get_mixed_set_dicts(
                    im_paths=im_paths, skip_no_pairs=args.skip_no_pairs
                ),
            )
            MetadataCatalog.get(f"mixed_{name}").thing_classes = list(CLASSES.keys())
    else:
        raise Exception("unsupported dataset!")

    log_every_n_seconds(logging.INFO, f"Successfully loaded {dataset_name} dataset")

    # Configure Model
    cfg = setup_cfg(
        arch=args.arch,
        dataset_name=dataset_name,
        batch_size=args.batch_size,
        epochs=args.epochs,
        eval_every_n_epochs=args.eval_every_n_epochs,
        base_lr=args.base_lr,
        decay_lr=args.decay_lr,
        output_dir=args.output_dir,
    )
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    log_every_n_seconds(
        logging.INFO,
        f"training for {cfg.SOLVER.MAX_ITER} iters, "
        + f"eval every {cfg.TEST.CUSTOM_EVAL_PERIOD}",
    )

    log_every_n_seconds(
        logging.INFO,
        f"batch size: {cfg.SOLVER.IMS_PER_BATCH}, base_lr: {cfg.SOLVER.BASE_LR}, "
        + f"milestones: {cfg.SOLVER.STEPS}, output_dir: {cfg.OUTPUT_DIR}",
    )

    # Instantiate CustomTrainer with COCO dataloaders, validation loss loop,
    # and best model checkpointing
    trainer = CustomTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    # EVALUATE model on TEST set using COCO style evaluation
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_best.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    evaluator = COCOEvaluator(
        cfg.DATASETS.TEST[1],
        output_dir=cfg.OUTPUT_DIR,
        use_fast_impl=False,
        allow_cached_coco=False,
    )
    test_loader = trainer.build_test_loader(cfg)
    predictor = DefaultPredictor(cfg)
    # write out results
    with open(os.path.join(cfg.OUTPUT_DIR, "test_results.txt"), "w") as f:
        results = inference_on_dataset(predictor.model, test_loader, evaluator)
        print(results)
        f.write(str(results))


if __name__ == "__main__":
    main()
