from base64 import decode
from typing import List
from detectron2.config.instantiate import instantiate
from detectron2.config.lazy import LazyConfig
from detectron2.data.catalog import DatasetCatalog, MetadataCatalog
from detectron2.evaluation.coco_evaluation import COCOEvaluator
from detectron2.evaluation.evaluator import inference_on_dataset
from detectron2.evaluation.testing import print_csv_format
from detectron2.utils.logger import setup_logger

import numpy as np
from strhub.data.module import SceneTextDataModule
import torch
import torch.nn as nn
from torchvision import transforms
import cv2
from PIL import ImageOps, Image

from detectron2 import model_zoo
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.structures import Instances
from detectron2.config import get_cfg
import detectron2.data.transforms as T
from detectron2.utils.visualizer import Visualizer, ColorMode
from data_utils import read_split_file, register_dataset


from parseq.strhub.models.parseq.system import PARSeq


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
    base = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
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


class ElevatorDetectorLazyConf(nn.Module):
    """
    Full pipeline of images

    Args:
        nn (_type_): _description_
    """

    def __init__(self, cfg, recovery_weights=None) -> None:
        super().__init__()
        self.recovery_weights = recovery_weights
        self.device = cfg.train.device

        # load instance segmentation model
        self.model = instantiate(cfg.model)
        self.model.to(self.device)
        self.model.eval()
        DetectionCheckpointer(self.model).load(cfg.train.finetuned_weights)

        # load recovery model
        if self.recovery_weights is not None:
            self.recovery_model = instantiate(cfg.model)
            self.recovery_model.to(cfg.train.device)
            self.recovery_model.eval()
            DetectionCheckpointer(self.recovery_model).load(self.recovery_weights)

        # load str model
        self.str = PARSeq.load_from_checkpoint("models/str_v1/parseq_str.ckpt")

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
        # original_img = cv2.imread(imgs[0]["img_path"])
        with torch.no_grad():
            outputs = self.model(imgs)
            detections = outputs[0]["instances"]
            if self.recovery_weights:
                class_map = self._convert_instances_to_class_map(detections)
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
                detections = Instances.cat([detections, recovered_instances])

            ### STR STUFF
            # labels = detections[detections.pred_classes == 0]
            # labels_expand = (
            #     torch.unsqueeze(labels.pred_masks, 1).expand(-1, 3, -1, -1) * 255
            # )

            # img_repeat = (
            #     torch.from_numpy(np.transpose(original_img, (2, 0, 1)))
            #     .to(self.device)
            #     .unsqueeze(0)
            #     .repeat(labels_expand.shape[0], 1, 1, 1)
            # )

            # iso_labels = torch.bitwise_and(labels_expand, img_repeat)

            # label_batch = torch.zeros((labels_expand.shape[0], 3, 32, 128))
            # img_transform = SceneTextDataModule.get_transform(self.str.hparams.img_size)

            # print(len(iso_labels))
            # print(len(labels.pred_boxes))
            # for i, (l, bbox) in enumerate(zip(iso_labels, labels.pred_boxes)):
            #     x1, y1, x2, y2 = bbox.type(torch.int32)
            #     label_crop = l[:, y1:y2, x1:x2].type(torch.uint8)
            #     tensor_to_pil = transforms.Compose([transforms.ToPILImage()])
            #     label_crop = tensor_to_pil(label_crop)
            #     x = img_transform(label_crop).unsqueeze(0)
            #     logits = self.str(x)
            #     pred = logits.softmax(-1)
            #     res, _ = self.str.tokenizer.decode(pred)
            #     label_crop.save(f"label_reading/{res[0]}.jpg")

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


if __name__ == "__main__":
    # resize_aug = T.ResizeShortestEdge(short_edge_length=1024, max_size=1024)
    # img_path = "data/panels/mixed/mixed_12.jpg"
    # original_img = cv2.imread(img_path)
    # height, width = original_img.shape[:2]
    # img = resize_aug.get_transform(original_img).apply_image(original_img)
    # img_tensor = torch.as_tensor(img.astype("float32").transpose(2, 0, 1))

    # input = {
    #     "image": img_tensor,
    #     "height": height,
    #     "width": width,
    #     "img_path": img_path,
    # }

    setup_logger()

    cfg = LazyConfig.load("configs/mask_rcnn_vit_base.py")
    pipeline = ElevatorDetectorLazyConf(
        cfg, recovery_weights="models/recovery_v1/model_best.pth"
    )
    # outputs = pipeline([input])[0]

    # v = Visualizer(
    #     original_img[:, :, ::-1],
    #     scale=1.0,
    #     instance_mode=ColorMode.SEGMENTATION,
    # )
    # out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    # cv2.imwrite("output.jpg", out.get_image()[:, :, ::-1])

    mixed_sets = read_split_file("data/panels/mixed/split.txt")

    # Register mixed datasets
    for spl, im_paths in zip(["train", "val", "test"], mixed_sets):
        DatasetCatalog.register(
            f"mixed_{spl}", lambda im_paths=im_paths: register_dataset(im_paths)
        )
        MetadataCatalog.get(f"mixed_{spl}").set(
            thing_classes=["label", "button"], thing_colors=[(0, 255, 0), (0, 0, 255)]
        )

    testloader_cfg = cfg.dataloader.mixed_test
    testloader = instantiate(testloader_cfg)
    evaluator = COCOEvaluator(
        dataset_name=testloader_cfg.dataset.names,
        use_fast_impl=False,
        allow_cached_coco=False,
        max_dets_per_image=200,
        output_dir="pipeline_test",
    )
    ret = inference_on_dataset(pipeline, testloader, evaluator)
    print_csv_format(ret)
