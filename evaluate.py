import os
import random

from termcolor import cprint
import cv2
from matplotlib import pyplot as plt

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import (
    DatasetCatalog,
    MetadataCatalog,
    get_detection_dataset_dicts,
    build_detection_test_loader,
)
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.visualizer import Visualizer, ColorMode

from data_utils import read_split_file, register_dataset

im_paths = "data/missed_detections/mixed/split.txt"
datasets = read_split_file(im_paths)
test_im_paths = datasets[-1]

DatasetCatalog.register(
    "test", func=lambda im_paths=test_im_paths: register_dataset(im_paths=im_paths)
)
MetadataCatalog.get("test").thing_classes = ["label", "button"]
metadata = MetadataCatalog.get("test")

cprint(f"loading testset...", "green")
testset = get_detection_dataset_dicts("test", filter_empty=False)
cprint(f"testset size: {len(testset)}", "green")

cfg = get_cfg()
cfg.merge_from_file(
    model_zoo.get_config_file("Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml")
)
cfg.MODEL.WEIGHTS = "models/cascade_mrcnn_recovery/model_best.pth"
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False
cfg.DATALOADER.NUM_WORKERS = 8
cfg.INPUT.RANDOM_FLIP = "none"
cfg.DATASETS.TEST = ("test",)

predictor = DefaultPredictor(cfg)


save_dir = "test_images/cascade_mrcnn_dect"
os.makedirs(save_dir, exist_ok=True)
for d in random.sample(testset, int(0.25 * len(testset))):
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    v = Visualizer(
        im[:, :, ::-1], metadata=metadata, instance_mode=ColorMode.SEGMENTATION
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imwrite(
        os.path.join(save_dir, os.path.basename(d["file_name"])),
        out.get_image()[:, :, ::-1],
    )

cprint("saved all images", "blue")
testloader = build_detection_test_loader(cfg, "test")
evaluator = COCOEvaluator(
    "test", output_dir="./eval_results", use_fast_impl=False, allow_cached_coco=False
)
print(inference_on_dataset(predictor.model, testloader, evaluator))
