from typing import List

import torch
import torch.nn as nn

from detectron2 import model_zoo
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.structures import Instances
from detectron2.config import get_cfg


def load_inference_model(options: dict) -> nn.Module:
    """
    Load a model for inference mode

    Args:
        arch (str): either mask_rcnn or cascade_mask_rcnn
        weights (str): path to the weights file
        score_thresh (float): threshold for predictions
        device (str): cuda or cpu

    Returns:
        nn.Module: model with weights loaded and appropriate parameters set
    """
    cfg = get_cfg()
    base = (
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        if options.get("arch") == "mask_rcnn"
        else "Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml"
    )
    cfg.merge_from_file(model_zoo.get_config_file(base))
    cfg.MODEL.WEIGHTS = options.get("weights")
    cfg.MODEL.DEVICE = options.get("device")
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = options.get("score_thresh")
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False

    # build model from config and load weights
    model = build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    model.eval()  # inference only
    return model


class ElevatorDetector(nn.Module):
    """
    Full pipeline of images

    Args:
        nn (_type_): _description_
    """

    def __init__(
        self,
        instance_segmentation_cfg: dict,
        recovery_cfg: dict = None,
        use_recovery=True,
        device=None,
    ) -> None:
        super().__init__()
        self.use_recovery = use_recovery
        self.device = device
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # add device to configs so both models are on the correct device
        instance_segmentation_cfg["device"] = self.device

        # load models in inference mode
        self.seg_model = load_inference_model(instance_segmentation_cfg)
        if use_recovery:
            recovery_cfg["device"] = self.device
            self.recovery_model = load_inference_model(recovery_cfg)

    def forward(self, imgs: List[dict]) -> List[dict]:
        """
        This model should only be evaluated and thus process one image at a time
        Detectron2 dataloaders return a list of dicts, for testloaders, there will be
        one dict in this list

        Args:
            imgs (List[dict]): list of Detectron2 model input dicts

        Returns:
            _type_: Detectron2 model output dict containing an Instances object
        """
        assert len(imgs) == 1
        # save original image height and width
        height, width = imgs[0]["height"], imgs[0]["width"]
        with torch.no_grad():
            outputs = self.seg_model(imgs)
            init_instances = outputs[0]["instances"]
            if not self.use_recovery:
                return [{"instances": init_instances}]
            else:
                class_map = self._convert_instances_to_class_map(init_instances)
                class_map = torch.permute(class_map, (2, 0, 1))  # H, W, C -> C, H, W
                # wrap in dictionary for Detectron2 model input
                class_map_dict = {
                    "height": height,
                    "width": width,
                    "image": class_map,
                }
                outputs = self.recovery_model([class_map_dict])
                recovered_instances = outputs[0]["instances"]
                # merge both detections into one output
                detections = Instances.cat([init_instances, recovered_instances])

                return [{"instances": detections}]

    def _convert_instances_to_class_map(self, x: Instances) -> torch.Tensor:
        # 0 -> labels, 1 -> buttons
        labels = x[x.pred_classes == 0]
        buttons = x[x.pred_classes == 1]
        class_map = torch.zeros(
            (*x.image_size, 3), dtype=torch.uint8, device=self.device
        )
        for features, color in zip([labels, buttons], [[0, 255, 0], [0, 0, 255]]):
            features = (
                torch.unsqueeze(features.pred_masks, -1)
                .expand(-1, -1, -1, 3)
                .sum(dim=0)
                * 255
            )
            features_rgb = torch.where(
                features == 255,
                torch.tensor(color, device=self.device, dtype=torch.uint8),
                torch.zeros((3,), device=self.device, dtype=torch.uint8),
            )
            torch.add(class_map, features_rgb, out=class_map)

        return class_map