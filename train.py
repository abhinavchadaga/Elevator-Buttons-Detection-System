import os
import logging
import time
import datetime

import numpy as np
from termcolor import cprint
import torch
import cv2

from detectron2.config.lazy import LazyConfig
from detectron2.config import instantiate
from detectron2.data import DatasetCatalog, MetadataCatalog, get_detection_dataset_dicts
import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import hooks, AMPTrainer, SimpleTrainer, default_writers, launch
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, print_csv_format
from detectron2.engine.defaults import create_ddp_model
from detectron2.utils import comm
from detectron2.utils.logger import log_every_n, setup_logger, log_every_n_seconds
from detectron2.utils.visualizer import Visualizer, ColorMode

from data_utils import read_split_file, register_dataset

setup_logger()


class Predictor:
    def __init__(self, cfg) -> None:
        self.model = instantiate(cfg.model)
        self.model.to(cfg.train.device)
        self.model.eval()
        self.metadata = MetadataCatalog.get(cfg.dataloader.train.dataset.names[0])

        DetectionCheckpointer(self.model).load(cfg.train.finetuned_weights)
        self.aug = T.ResizeShortestEdge(
            short_edge_length=cfg.dataloader.image_size,
            max_size=cfg.dataloader.image_size,
        )

        self.input_format = "BGR"

    def __call__(self, original_image) -> dict:
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
        """
        with torch.no_grad():
            height, width = original_image.shape[:2]
            img = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(img.astype("float32").transpose(2, 0, 1))

            inputs = {"image": image, "height": height, "width": width}
            predictions = self.model([inputs])[0]
            return predictions


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


def do_train(cfg):
    # LOAD MODEL
    model = instantiate(cfg.model)
    logger = logging.getLogger("detectron2")
    logger.info("Model:\n{}".format(model))
    model.to(cfg.train.device)
    model = create_ddp_model(model, **cfg.train.ddp)

    # LOAD OPTIMIZER
    cfg.optimizer.params.base_lr = 0.00025
    cfg.optimizer.lr = 0.00025
    cfg.optimizer.params.model = model
    optim = instantiate(cfg.optimizer)

    # LOAD TRAINLOADER
    trainloader = instantiate(cfg.dataloader.train)

    # CREATE TRAINER AND REGISTER HOOKS
    trainer = (AMPTrainer if cfg.train.amp.enabled else SimpleTrainer)(
        model, trainloader, optim
    )
    checkpointer = DetectionCheckpointer(model, cfg.train.output_dir, trainer=trainer)
    trainer.register_hooks(
        [
            hooks.EvalHook(
                cfg.train.eval_period,
                lambda: do_val(model, instantiate(cfg.dataloader.val)),
                eval_after_train=True,
            ),
            hooks.IterationTimer(),
            hooks.LRScheduler(scheduler=instantiate(cfg.lr_multiplier)),
            hooks.BestCheckpointer(
                cfg.train.eval_period, checkpointer, "validation_loss", "min"
            )
            if comm.is_main_process()
            else None,
            hooks.PeriodicWriter(
                default_writers(cfg.train.output_dir, cfg.train.max_iter),
                period=cfg.train.log_period,
            )
            if comm.is_main_process()
            else None,
        ]
    )

    # TRAIN MODEL
    checkpointer.load(path=cfg.train.init_checkpoint)
    trainer.train(0, cfg.train.max_iter)


def do_test(cfg, dataset: str, score_thresh=0.7, save_images=True):
    assert dataset in ("mixed", "ut_west_campus")
    cfg.model.roi_heads.box_predictor.test_score_thresh = score_thresh
    testloader_cfg = (
        cfg.dataloader.mixed_test
        if dataset == "mixed"
        else cfg.dataloader.ut_west_campus_test
    )
    predictor = Predictor(cfg)
    testloader = instantiate(testloader_cfg)
    evaluator = COCOEvaluator(
        dataset_name=testloader_cfg.dataset.names,
        use_fast_impl=False,
        allow_cached_coco=False,
        max_dets_per_image=200,
        output_dir=os.path.join(cfg.train.output_dir, "inference", dataset),
    )
    ret = inference_on_dataset(predictor.model, testloader, evaluator)
    print_csv_format(ret)

    if save_images:
        log_every_n_seconds(logging.INFO, "saving test images....")
        testset = get_detection_dataset_dicts(f"{dataset}_test", filter_empty=False)
        save_dir = os.path.join(cfg.train.output_dir, f"{dataset}_test_images")
        os.makedirs(save_dir, exist_ok=True)
        for d in testset:
            img = cv2.imread(d["file_name"])
            outputs = predictor(img)
            v = Visualizer(
                img[:, :, ::-1],
                metadata=MetadataCatalog.get(f"{dataset}_test"),
                scale=1.0,
                instance_mode=ColorMode.SEGMENTATION,
            )
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            cv2.imwrite(
                os.path.join(save_dir, f"{os.path.basename(d['file_name'])}"),
                out.get_image()[:, :, ::-1],
            )

    return ret


def main():
    # REGISTER NEW DATASETS
    log_every_n(logging.INFO, "(SEGMENTATION MODEL) registering datasets...")
    mixed_sets = read_split_file("data/panels/mixed/split.txt")
    ut_west_campus_sets = read_split_file("data/panels/ut_west_campus/split.txt")

    # Register mixed datasets
    for spl, im_paths in zip(["train", "val", "test"], mixed_sets):
        DatasetCatalog.register(
            f"mixed_{spl}", lambda im_paths=im_paths: register_dataset(im_paths)
        )
        MetadataCatalog.get(f"mixed_{spl}").set(
            thing_classes=["label", "button"], thing_colors=[(0, 255, 0), (0, 0, 255)]
        )

    # Register ut_west_campus datasets
    for spl, im_paths in zip(["train", "val", "test"], ut_west_campus_sets):
        DatasetCatalog.register(
            f"ut_west_campus_{spl}",
            lambda im_paths=im_paths: register_dataset(im_paths),
        )
        MetadataCatalog.get(f"ut_west_campus_{spl}").set(
            thing_classes=["label", "button"], thing_colors=[(0, 255, 0), (0, 0, 255)]
        )

    # load config and train model
    cfg = LazyConfig.load("configs/mask_rcnn_vit_base.py")
    # do_train(cfg)
    do_test(cfg, "mixed")
    do_test(cfg, "ut_west_campus")


if __name__ == "__main__":
    launch(main, num_gpus_per_machine=1, num_machines=1, machine_rank=0, dist_url=None)
