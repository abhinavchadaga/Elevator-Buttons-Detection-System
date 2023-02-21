import os
import random
from detectron2.checkpoint.detection_checkpoint import DetectionCheckpointer

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
from detectron2.config import LazyConfig, instantiate

from data_utils import read_split_file, register_dataset
import mixed_dataloaders

im_paths = "data/panels/mixed/split.txt"
datasets = read_split_file(im_paths)
test_im_paths = datasets[-1]

DatasetCatalog.register(
    "mixed_test",
    func=lambda im_paths=test_im_paths: register_dataset(im_paths=im_paths),
)
MetadataCatalog.get("mixed_test").thing_classes = ["label", "button"]
metadata = MetadataCatalog.get("mixed_test")

cprint(f"loading testset...", "green")
testset = get_detection_dataset_dicts("mixed_test", filter_empty=False)
cprint(f"testset size: {len(testset)}", "green")

cfg = model_zoo.get_config("new_baselines/mask_rcnn_R_101_FPN_400ep_LSJ.py")
cfg.dataloader = mixed_dataloaders.dataloader
cfg.model.backbone.bottom_up.stem.norm = "BN"
cfg.model.backbone.bottom_up.stages.norm = "BN"
cfg.model.backbone.norm = "BN"
cfg.model.roi_heads.num_classes = 2
cfg.model.roi_heads.box_predictor.test_score_thresh = 0.7


model = instantiate(cfg.model)
DetectionCheckpointer(model).load("models/mask_rcnn_res101/model_best.pth")

# for d in testset:
#     im = cv2.imread(d["file_name"])
#     outputs = predictor(im)
#     v = Visualizer(
#         im[:, :, ::-1], metadata=metadata, instance_mode=ColorMode.SEGMENTATION
#     )
#     out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
#     cv2.imwrite(
#         os.path.join(save_dir, os.path.basename(d["file_name"])),
#         out.get_image()[:, :, ::-1],
#     )

# cprint("saved all images", "blue")
evaluator = instantiate(cfg.dataloader.evaluator)

print(inference_on_dataset(model, instantiate(cfg.dataloader.test), evaluator))
