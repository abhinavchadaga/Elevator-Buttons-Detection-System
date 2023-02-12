import argparse
import os
from unittest import TestLoader

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
from detectron2.engine import DefaultTrainer, BestCheckpointer, DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

from data_utils import random_split_mixed_set, get_mixed_set_dicts, CLASSES


class CustomTrainer(DefaultTrainer):
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
            total_batch_size=2,
            num_workers=4,
        )

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(
            dataset=get_detection_dataset_dicts(cfg.DATASETS.TEST, filter_empty=False),
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


def setup_cfg(dataset_name: str, batch_size=2, base_lr=0.00025) -> CfgNode:
    cmrcnn = "Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml"
    mrcnn = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(cmrcnn))
    cfg.INPUT.RANDOM_FLIP = "none"  # turn off default augmentation
    cfg.DATASETS.TRAIN = (f"{dataset_name}_train",)
    cfg.DATASETS.TEST = (f"{dataset_name}_val",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(cmrcnn)
    cfg.SOLVER.IMS_PER_BATCH = batch_size
    cfg.SOLVER.BASE_LR = base_lr
    cfg.TEST.EVAL_PERIOD = 200
    cfg.SOLVER.MAX_ITER = 2625  # equivalent to 75 epochs for the mixed dataset
    cfg.SOLVER.STEPS = []  # do not decay learning rate
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    cfg.OUTPUT_DIR = "cmrcnn_75ep_coco_dataloader"
    return cfg


def main():
    parser = argparse.ArgumentParser(
        description="training script for Detectron2 models"
    )
    parser.add_argument("--dataset", "-d", type=str, default="mixed")
    args = parser.parse_args()
    dataset_name = args.dataset

    if dataset_name == "mixed":
        # split and register mixed dataset
        img_paths = random_split_mixed_set("data/panels/mixed")
        for name, paths in zip(["train", "val", "test"], img_paths):
            DatasetCatalog.register(
                name=f"mixed_{name}",
                func=lambda im_paths=paths: get_mixed_set_dicts(im_paths=im_paths),
            )
            MetadataCatalog.get(f"mixed_{name}").thing_classes = list(CLASSES.keys())
    else:
        raise Exception("unsupported dataset!")

    # TRAIN on TRAIN and VAL
    cfg = setup_cfg(dataset_name=dataset_name)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = CustomTrainer(cfg)
    trainer.register_hooks(
        [
            BestCheckpointer(
                eval_period=cfg.TEST.EVAL_PERIOD,
                checkpointer=trainer.checkpointer,
                val_metric="bbox/AP50",
                mode="max",
            )
        ]
    )
    trainer.resume_or_load(resume=False)
    trainer.train()

    # EVALUATE on TEST
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_best.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
    cfg.DATASETS.TEST = f"{dataset_name}_test"
    evaluator = COCOEvaluator(
        cfg.DATASETS.TEST,
        output_dir=cfg.OUTPUT_DIR,
        use_fast_impl=False,
        allow_cached_coco=False,
    )
    test_loader = trainer.build_test_loader(cfg, cfg.DATASETS.TEST)
    predictor = DefaultPredictor(cfg)
    with open(os.path.join(cfg.OUTPUT_DIR, "test_results.txt"), "w") as f:
        results = inference_on_dataset(predictor.model, test_loader, evaluator)
        print(results)
        f.write(str(results))


if __name__ == "__main__":
    main()
