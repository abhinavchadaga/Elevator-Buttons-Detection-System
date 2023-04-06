from base64 import decode
from fileinput import filename
import json
import logging
import math
import os
from time import time
from typing import List
from detectron2.config.instantiate import instantiate
from detectron2.config.lazy import LazyConfig
from detectron2.data import DatasetCatalog, MetadataCatalog, get_detection_dataset_dicts
from detectron2.data.build import build_detection_test_loader
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.detection_utils import annotations_to_instances
from detectron2.evaluation.coco_evaluation import COCOEvaluator
from detectron2.evaluation.evaluator import inference_on_dataset
from detectron2.evaluation.testing import print_csv_format
from detectron2.structures.boxes import Boxes
from detectron2.structures import pairwise_iou
from detectron2.utils.logger import log_every_n, setup_logger

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
from tqdm import tqdm
from data_utils import read_split_file, register_dataset

from train import Predictor


from strhub.models.parseq.system import PARSeq

symbol_map = {
    "<|>": "open",
    ">|<": "close",
    "^": "alarm",
    "&": "call",
    "#": "stop",
    "><": "rear door close",
    "<>": "rear door open",
    "F|>": "front door open",
    "F|<": "front door close",
    "R|>": "rear door open",
    "R|<": "rear door close",
}


def calc_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


class ElevatorDetector(nn.Module):
    """
    Full pipeline of images

    Args:
        nn (_type_): _description_
    """

    def __init__(self, use_recovery=True) -> None:
        super().__init__()
        self.use_recovery = use_recovery
        self.device = "cuda"
        cfg = get_cfg()
        base = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
        cfg.merge_from_file(model_zoo.get_config_file(base))
        cfg.MODEL.WEIGHTS = "models/segmentation_resnet/model_best.pth"
        cfg.MODEL.DEVICE = self.device
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False
        self.cfg = cfg

        #  Load segmentation model
        self.model = build_model(cfg)
        DetectionCheckpointer(self.model).load(cfg.MODEL.WEIGHTS)
        self.model.eval()

        if self.use_recovery:
            cfg.MODEL.WEIGHTS = "models/recovery_resnet/model_best.pth"
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9
            self.recovery_model = build_model(cfg)
            DetectionCheckpointer(self.recovery_model).load(cfg.MODEL.WEIGHTS)
            self.recovery_model.eval()

        # load str model
        self.str = PARSeq.load_from_checkpoint("models/str_v3_combined/parseq_str.ckpt")

    def forward(self, imgs: List[dict]):
        # save original image height and width
        height, width = imgs[0]["height"], imgs[0]["width"]
        with torch.no_grad():
            outputs = self.model(imgs)
            detections = outputs[0]["instances"].to(self.device)
            if self.use_recovery:
                class_map = self._convert_instances_to_class_map(detections)
                class_map = torch.permute(class_map, (2, 0, 1))  # H, W, C -> C, H, W
                # wrap in dictionary for Detectron2 model input
                class_map_dict = {
                    "height": height,
                    "width": width,
                    "image": class_map,
                }
                outputs = self.recovery_model([class_map_dict])
                recovered_instances = outputs[0]["instances"].to(self.device)
                # merge both detections into one output
                detections = Instances.cat([detections, recovered_instances])

            return [{"instances": detections}]

    def run(
        self, imgs: List[dict], floor: str = None, floors: set = None
    ) -> List[dict]:
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
        original_img = Image.open(imgs[0]["img_path"])
        original_img = ImageOps.exif_transpose(original_img)
        filename = os.path.basename(imgs[0]["img_path"])
        floor = floor.rstrip().replace(" ", "_").lower()
        with torch.no_grad():
            outputs = self.model(imgs)
            detections = outputs[0]["instances"].to(self.device)
            if self.use_recovery:
                class_map = self._convert_instances_to_class_map(detections)
                class_map = torch.permute(class_map, (2, 0, 1))  # H, W, C -> C, H, W
                # wrap in dictionary for Detectron2 model input
                class_map_dict = {
                    "height": height,
                    "width": width,
                    "image": class_map,
                }
                outputs = self.recovery_model([class_map_dict])
                recovered_instances = outputs[0]["instances"].to(self.device)
                # merge both detections into one output
                detections = Instances.cat([detections, recovered_instances])

            labels = detections[detections.pred_classes == 0]
            target_label = None
            img_transform = SceneTextDataModule.get_transform(self.str.hparams.img_size)
            os.makedirs("label_reading", exist_ok=True)
            for i, bbox in enumerate(labels.pred_boxes):
                bbox = bbox.tolist()
                bbox = [int(x) for x in bbox]
                label_crop = original_img.crop(box=(bbox)).convert("RGB")
                label_crop = ImageOps.pad(label_crop, size=(64, 64))
                label_crop.save(f"label_reading/{filename.split('.')[0]}-{i}.jpg")
                input = img_transform(label_crop).unsqueeze(0)
                logits = self.str(input)
                pred = logits.softmax(-1)
                res, _ = self.str.tokenizer.decode(pred)
                if res[0].rstrip().lower().replace("*", "") == floor.replace("*", ""):
                    target_label = labels[i]
                    break

            if target_label is None:
                print("could not find ", floor, "!!!")
                return None

            # GET BUTTON
            buttons = detections[detections.pred_classes == 1]
            label_bbox = target_label.pred_boxes[0]
            label_center = label_bbox.get_centers()[0].tolist()
            label_bbox = label_bbox.tensor.tolist()[0]
            global_target_button_idx = None
            target_button_idx = None
            min_right = float("inf")
            min_global = float("inf")
            for i, btn_center in enumerate(buttons.pred_boxes.get_centers()):
                dist = math.dist(btn_center, label_center)
                if (
                    btn_center[0] > label_center[0]
                    and btn_center[1] < label_bbox[3]
                    and btn_center[1] > label_bbox[1]
                    and dist < min_right
                ):
                    min_right = dist
                    target_button_idx = i
                elif dist < min_global:
                    min_global = dist
                    global_target_button_idx = i

            target_button = (
                buttons[target_button_idx]
                if target_button_idx is not None
                else buttons[global_target_button_idx]
            )

            return [{"target_label": target_label, "target_button": target_button}]

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
            # cfg.model.roi_heads.box_predictor.test_score_thresh = 0.7
            self.recovery_model = instantiate(cfg.model)
            self.recovery_model.to(self.device)
            self.recovery_model.eval()
            DetectionCheckpointer(self.recovery_model).load(self.recovery_weights)

        # load str model
        self.str = PARSeq.load_from_checkpoint("models/str_v3_combined/parseq_str.ckpt")
        # self.str = torch.hub.load("baudm/parseq", "parseq", pretrained=True).eval()

    def forward(self, imgs: List[dict]):
        assert len(imgs) == 1
        # save original image height and width
        height, width = imgs[0]["height"], imgs[0]["width"]
        with torch.no_grad():
            outputs = self.model(imgs)
            detections = outputs[0]["instances"].to(self.device)
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
                recovered_instances = outputs[0]["instances"].to(self.device)
                # merge both detections into one output
                detections = Instances.cat([detections, recovered_instances])

            return [{"instances": detections}]

    def run(
        self, imgs: List[dict], floor: str = None, floors: set = None
    ) -> List[dict]:
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
        original_img = Image.open(imgs[0]["img_path"])
        original_img = ImageOps.exif_transpose(original_img)
        filename = os.path.basename(imgs[0]["img_path"])
        default_colors = MetadataCatalog.get("mixed_train")
        burnt_orange = MetadataCatalog.get("mixed_test")
        floor = floor.rstrip().replace(" ", "_").lower()
        with torch.no_grad():
            outputs = self.model(imgs)
            detections = outputs[0]["instances"].to(self.device)

            # write initial detections out to disk
            visualizer = Visualizer(
                np.asarray(original_img),
                metadata=default_colors,
                scale=1,
                instance_mode=ColorMode.SEGMENTATION,
            )
            out = visualizer.draw_instance_predictions(detections.to("cpu"))
            cv2.imwrite(f"init_detections.png", out.get_image()[:, :, ::-1])

            if self.recovery_weights:
                class_map = self._convert_instances_to_class_map(detections)
                og_class_map = class_map.to("cpu").numpy()
                cv2.imwrite("init_class_map.png", og_class_map[:, :, ::-1])
                class_map = torch.permute(class_map, (2, 0, 1))  # H, W, C -> C, H, W
                # wrap in dictionary for Detectron2 model input
                class_map_dict = {
                    "height": height,
                    "width": width,
                    "image": class_map,
                }
                outputs = self.recovery_model([class_map_dict])
                visualizer = Visualizer(
                    og_class_map,
                    metadata=burnt_orange,
                    scale=1,
                    instance_mode=ColorMode.SEGMENTATION,
                )
                out = visualizer.draw_instance_predictions(
                    outputs[0]["instances"].to("cpu")
                )
                cv2.imwrite("final_class_map.png", out.get_image()[:, :, ::-1])
                recovered_instances = outputs[0]["instances"].to(self.device)
                # merge both detections into one output
                detections = Instances.cat([detections, recovered_instances])
                visualizer = Visualizer(
                    np.asarray(original_img),
                    metadata=default_colors,
                    scale=1,
                    instance_mode=ColorMode.SEGMENTATION,
                )
                out = visualizer.draw_instance_predictions(detections.to("cpu"))
                cv2.imwrite(f"with_missed_detections.png", out.get_image()[:, :, ::-1])

            labels = detections[detections.pred_classes == 0]
            target_label = None
            img_transform = SceneTextDataModule.get_transform(self.str.hparams.img_size)
            os.makedirs("label_reading", exist_ok=True)
            for i, bbox in enumerate(labels.pred_boxes):
                bbox = bbox.tolist()
                bbox = [int(x) for x in bbox]
                label_crop = original_img.crop(box=(bbox)).convert("RGB")
                label_crop = ImageOps.pad(label_crop, size=(64, 64))
                label_crop.save(f"label_reading/{filename.split('.')[0]}-{i}.jpg")
                input = img_transform(label_crop).unsqueeze(0)
                logits = self.str(input)
                pred = logits.softmax(-1)
                res, _ = self.str.tokenizer.decode(pred)
                if res[0].rstrip().lower().replace("*", "") == floor.replace("*", ""):
                    target_label = labels[i]
                    break

            if target_label is None:
                print("could not find ", floor, "!!!")
                return None

            # GET BUTTON
            buttons = detections[detections.pred_classes == 1]
            label_bbox = target_label.pred_boxes[0]
            label_center = label_bbox.get_centers()[0].tolist()
            label_bbox = label_bbox.tensor.tolist()[0]

            # DRAW LABEL
            visualizer = Visualizer(
                np.asarray(original_img),
                metadata=burnt_orange,
                scale=1,
                instance_mode=ColorMode.SEGMENTATION,
            )
            visualizer.draw_instance_predictions(target_label.to("cpu"))
            out = visualizer.draw_text(
                floor,
                (
                    int(label_bbox[0] + label_bbox[2]) / 2,
                    int((label_bbox[1] + label_bbox[3]) / 2),
                ),
                color="r",
            )
            cv2.imwrite("identify_label.png", out.get_image()[:, :, ::-1])

            global_target_button_idx = None
            target_button_idx = None
            min_right = float("inf")
            min_global = float("inf")
            for i, btn_center in enumerate(buttons.pred_boxes.get_centers()):
                dist = math.dist(btn_center, label_center)
                if (
                    btn_center[0] > label_center[0]
                    and btn_center[1] < label_bbox[3]
                    and btn_center[1] > label_bbox[1]
                    and dist < min_right
                ):
                    min_right = dist
                    target_button_idx = i
                elif dist < min_global:
                    min_global = dist
                    global_target_button_idx = i

            target_button = (
                buttons[target_button_idx]
                if target_button_idx is not None
                else buttons[global_target_button_idx]
            )

            visualizer = Visualizer(
                np.asarray(original_img),
                metadata=burnt_orange,
                scale=1,
                instance_mode=ColorMode.SEGMENTATION,
            )
            visualizer.draw_instance_predictions(target_label.to("cpu"))
            visualizer.draw_instance_predictions(target_button.to("cpu"))
            target_btn_bbox = target_button.pred_boxes[0].tensor.tolist()[0]
            out = visualizer.draw_text(
                floor,
                (
                    int(target_btn_bbox[0] + target_btn_bbox[2]) / 2,
                    int((target_btn_bbox[1] + target_btn_bbox[3]) / 2),
                ),
                color="r",
            )
            cv2.imwrite("final_output.png", out.get_image()[:, :, ::-1])

            return [{"target_label": target_label, "target_button": target_button}]

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


def test_pipeline_acc(dataset):
    datasets = read_split_file(f"data/panels/{dataset}/split.txt")

    # Register mixed datasets
    for spl, im_paths in zip(["train", "val", "test"], datasets):
        DatasetCatalog.register(
            f"{dataset}_{spl}",
            lambda im_paths=im_paths: register_dataset(im_paths),
        )
        colors = [(0, 255, 0), (0, 0, 255)]
        if spl == "test":
            colors = [(191, 87, 0), (191, 87, 0)]
        MetadataCatalog.get(f"{dataset}_{spl}").set(
            thing_classes=["label", "button"], thing_colors=colors
        )
    metadata = MetadataCatalog.get(f"{dataset}_train")

    log_every_n(logging.INFO, "LOADING PIPELINE...")
    cfg = LazyConfig.load("configs/mask_rcnn_vit_base.py")
    pipeline = ElevatorDetectorLazyConf(
        cfg, recovery_weights="models/recovery_vit/model_best.pth"
    )

    # pipeline = ElevatorDetector(use_recovery=True)

    with open(f"data/panels/{dataset}/pipeline_gt.json") as f:
        gts = json.load(f)

    log_every_n(logging.INFO, "starting evaluation")
    resize_aug = T.ResizeShortestEdge(short_edge_length=1024, max_size=1024)
    ## stats ##
    correct = 0
    total = 0
    runtime = 0.0
    ########
    os.makedirs("pipeline_test_images", exist_ok=True)
    for filename, labels in gts.items():
        if "mixed_85.jpg" not in filename:
            continue
        local_correct, local_total, local_runtime = 0, 0, 0.0
        img_path = os.path.join(f"data/panels/{dataset}", filename)
        # img_path = "mixed_85.jpg"
        original_img = cv2.imread(img_path)
        height, width = original_img.shape[:2]
        img = resize_aug.get_transform(original_img).apply_image(original_img)
        img_tensor = torch.as_tensor(img.astype("float32").transpose(2, 0, 1))
        input = {
            "image": img_tensor,
            "height": height,
            "width": width,
            "img_path": img_path,
        }

        for l, gt_bbox in labels.items():
            if l != "31":
                continue
            # get prediction
            start = time()
            outputs = pipeline.run([input], l)
            finish = time()
            local_runtime += finish - start

            if outputs is None:
                local_total += 1
                cv2.imwrite(f"pipeline_test_images/{filename}_{l}.jpg", original_img)
                continue
            outputs = outputs[0]

            v = Visualizer(
                original_img[:, :, ::-1],
                scale=1.0,
                metadata=metadata,
                instance_mode=ColorMode.SEGMENTATION,
            )
            btn_bbox = outputs["target_button"].pred_boxes[0].to("cpu")
            btn_bbox = btn_bbox.tensor.tolist()[0]
            out = v.draw_instance_predictions(outputs["target_button"].to("cpu"))
            out = v.draw_instance_predictions(outputs["target_label"].to("cpu"))
            out = out.get_image()[:, :, ::-1].astype(np.uint8)
            cv2.putText(
                out,
                text=l if l not in symbol_map else symbol_map[l],
                org=(int(btn_bbox[0]) + 5, int((btn_bbox[1] + btn_bbox[3]) / 2)),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=2,
                color=(0, 255, 0),
                thickness=5,
            )
            cv2.imwrite(
                f"pipeline_test_images/{os.path.splitext(filename)[0]}_{l}.jpg", out
            )
            iou = calc_iou(btn_bbox, gt_bbox)
            if iou > 0.5:
                local_correct += 1
            else:
                log_every_n(logging.INFO, f"gt: {gt_bbox}, pred: {btn_bbox}")
                log_every_n(logging.INFO, f"{l} incorrect!! IoU: {iou}")

            local_total += 1

        correct += local_correct
        total += local_total
        runtime += local_runtime

        log_every_n(
            logging.INFO,
            f"completed {filename} with {local_correct / local_total * 100:.2f}% accuracy, {local_runtime / local_total:.2f} seconds per img",
        )
        break

    log_every_n(logging.INFO, f"correct: {correct}")
    log_every_n(logging.INFO, f"total: {total}")
    log_every_n(logging.INFO, f"final accuracy: {correct / total * 100:.2f}%")
    log_every_n(
        logging.INFO, f"avg inference speed: {runtime / total:.2f} seconds per image"
    )


def test_vit_segmentation(dataset_name: str):
    datasets = read_split_file(f"data/panels/{dataset_name}/split.txt")

    # Register mixed datasets
    for spl, im_paths in zip(["train", "val", "test"], datasets):
        DatasetCatalog.register(
            f"{dataset_name}_{spl}",
            lambda im_paths=im_paths: register_dataset(im_paths),
        )
        MetadataCatalog.get(f"{dataset_name}_{spl}").set(
            thing_classes=["label", "button"], thing_colors=[(0, 255, 0), (0, 0, 255)]
        )

    cfg = LazyConfig.load("configs/mask_rcnn_vit_base.py")
    pipeline = ElevatorDetectorLazyConf(
        cfg, recovery_weights="models/recovery_vit/model_best.pth"
    )
    # pipeline = ElevatorDetectorLazyConf(cfg)

    testloader_cfg = (
        cfg.dataloader.mixed_test
        if dataset_name == "mixed"
        else cfg.dataloader.ut_west_campus_test
    )
    testloader = instantiate(testloader_cfg)
    evaluator = COCOEvaluator(
        dataset_name=testloader_cfg.dataset.names,
        use_fast_impl=False,
        allow_cached_coco=False,
        max_dets_per_image=200,
        output_dir="./output",
    )
    ret = inference_on_dataset(pipeline, testloader, evaluator)
    print_csv_format(ret)


def test_resnet_segmentation(dataset_name: str):
    datasets = read_split_file(f"data/panels/{dataset_name}/split.txt")

    # Register mixed datasets
    for spl, im_paths in zip(["train", "val", "test"], datasets):
        DatasetCatalog.register(
            f"{dataset_name}_{spl}",
            lambda im_paths=im_paths: register_dataset(im_paths),
        )
        MetadataCatalog.get(f"{dataset_name}_{spl}").set(
            thing_classes=["label", "button"], thing_colors=[(0, 255, 0), (0, 0, 255)]
        )

    pipeline = ElevatorDetector(use_recovery=False)
    testloader = build_detection_test_loader(pipeline.cfg, f"{dataset_name}_test")
    evaluator = COCOEvaluator(
        dataset_name=f"{dataset_name}_test",
        use_fast_impl=False,
        allow_cached_coco=False,
        max_dets_per_image=200,
        output_dir="./output",
    )
    ret = inference_on_dataset(pipeline, testloader, evaluator)
    print_csv_format(ret)


def test_button_accuracy(
    dataset_name: str, backbone: str, use_recovery=True, ious=[0.5]
):
    if backbone == "vit":
        cfg = LazyConfig.load("configs/mask_rcnn_vit_base.py")
        recovery_weights = (
            "models/recovery_vit/model_best.pth" if use_recovery else None
        )
        pipeline = ElevatorDetectorLazyConf(cfg, recovery_weights=recovery_weights)
    else:
        pipeline = ElevatorDetector(use_recovery=use_recovery)

    datasets = read_split_file(f"data/panels/{dataset_name}/split.txt")

    # Register mixed datasets
    for spl, im_paths in zip(["train", "val", "test"], datasets):
        DatasetCatalog.register(
            f"{dataset_name}_{spl}",
            lambda im_paths=im_paths: register_dataset(im_paths),
        )
        MetadataCatalog.get(f"{dataset_name}_{spl}").set(
            thing_classes=["label", "button"], thing_colors=[(0, 255, 0), (0, 0, 255)]
        )
    metadata = MetadataCatalog.get(f"{dataset_name}_train")

    testset = get_detection_dataset_dicts(f"{dataset_name}_test", filter_empty=False)

    resize_aug = T.ResizeShortestEdge(short_edge_length=1024, max_size=1024)
    out = []
    for iou in ious:
        log_every_n(logging.INFO, f"IoU threshold: {iou}")
        labels_correct, btns_correct, total_labels, total_btns = 0, 0, 0, 0
        for d in tqdm(testset, desc="images"):
            gt_instances = annotations_to_instances(
                d["annotations"], (d["height"], d["width"])
            )
            original_img = cv2.imread(d["file_name"])
            height, width = original_img.shape[:2]
            img = resize_aug.get_transform(original_img).apply_image(original_img)
            img_tensor = torch.as_tensor(img.astype("float32").transpose(2, 0, 1))
            input = {
                "image": img_tensor,
                "height": height,
                "width": width,
                "img_path": d["file_name"],
            }
            predictions = pipeline([input])[0]["instances"].to("cpu")
            gt_labels = gt_instances[gt_instances.gt_classes == 0].gt_boxes
            gt_btns = gt_instances[gt_instances.gt_classes == 1].gt_boxes

            pred_labels = predictions[predictions.pred_classes == 0].pred_boxes
            pred_btns = predictions[predictions.pred_classes == 1].pred_boxes

            labels_iou = pairwise_iou(gt_labels, pred_labels)
            matches = torch.max(labels_iou, dim=1).values
            local_labels_correct = torch.where(matches > iou, 1, 0).sum()
            local_total_labels = len(gt_labels)

            btns_iou = pairwise_iou(gt_btns, pred_btns)
            matches = torch.max(btns_iou, dim=1).values
            local_btns_correct = torch.where(matches > iou, 1, 0).sum()
            local_total_btns = len(gt_btns)

            filename = os.path.basename(d["file_name"])
            log_every_n(
                logging.INFO,
                f"{filename}: {local_labels_correct / local_total_labels * 100:.2f}% label accuracy, {local_btns_correct / local_total_btns * 100:.2f}% btn accuracy",
            )

            labels_correct += local_labels_correct
            total_labels += local_total_labels
            btns_correct += local_btns_correct
            total_btns += local_total_btns

        log_every_n(logging.INFO, f"final metrics")
        log_every_n(
            logging.INFO,
            f"label accuracy  :  {labels_correct} / {total_labels} => {labels_correct / total_labels * 100:.2f}%",
        )
        log_every_n(
            logging.INFO,
            f"button accuracy : {btns_correct} / {total_btns} => {btns_correct / total_btns * 100:.2f}%",
        )
        log_every_n(
            logging.INFO,
            f"total accuracy :  {labels_correct + btns_correct} / {total_labels + total_btns} =>  {(labels_correct + btns_correct) / (total_labels + total_btns) * 100:.2f}%",
        )
        out.extend(
            [
                f"{dataset_name}, {backbone}, use_recovery={use_recovery}, iou={iou}\n\n"
                f"label accuracy  :  {labels_correct} / {total_labels} => {labels_correct / total_labels * 100:.2f}%\n",
                f"button accuracy : {btns_correct} / {total_btns} => {btns_correct / total_btns * 100:.2f}%\n",
                f"total accuracy :  {labels_correct + btns_correct} / {total_labels + total_btns} =>  {(labels_correct + btns_correct) / (total_labels + total_btns) * 100:.2f}%\n\n",
            ]
        )
    return out


if __name__ == "__main__":
    setup_logger()
    test_pipeline_acc("mixed")
    # test_vit_segmentation("mixed")
    # test_resnet_segmentation(dataset_name="mixed")
    # for iou in [0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9]:

    # with open("accuracy_results_2.txt", "a") as f:
    #     f.writelines(
    #         test_button_accuracy(
    #             "mixed",
    #             "vit",
    #             True,
    #             ious=[0.5],
    #         )
    #     )
