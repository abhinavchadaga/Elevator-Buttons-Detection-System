import os
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
import numpy as np
import torch

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.data.detection_utils import read_image

import cv2

cfg = LazyConfig.load(
    "detectron2/configs/new_baselines/mask_rcnn_R_101_FPN_400ep_LSJ.py"
)

# edit the config to utilize common Batch Norm
cfg.model.backbone.bottom_up.stem.norm = "BN"
cfg.model.backbone.bottom_up.stages.norm = "BN"
cfg.model.backbone.norm = "BN"
cfg.model.roi_heads.num_classes = 2
cfg.model.roi_heads.box_predictor.test_score_thresh = 0.7

model = instantiate(cfg.model)

DetectionCheckpointer(model).load(
    "models/mask_rcnn_res101/model_best.pth"
)  # load a file, usually from cfg.MODEL.WEIGHTS

og_img = cv2.imread("data/panels/mixed/72.jpg")

# read image for inference input
# use PIL, to be consistent with evaluation
img = torch.from_numpy(
    np.ascontiguousarray(read_image("data/panels/mixed/72.jpg", format="BGR"))
)
img = img.permute(2, 0, 1)  # HWC -> CHW
if torch.cuda.is_available():
    img = img.cuda()
inputs = [{"image": img}]

# run the model
model.eval()
with torch.no_grad():
    predictions_ls = model(inputs)
predictions = predictions_ls[0]
print(predictions["instances"].get("scores"))

v = Visualizer(
    og_img[:, :, ::-1],
    instance_mode=ColorMode.SEGMENTATION,
)
out = v.draw_instance_predictions(predictions["instances"].to("cpu"))
cv2.imwrite(
    "output.jpg",
    out.get_image()[:, :, ::-1],
)
