import json
from turtle import position
from typing import List, Tuple
import math
from glob import glob
import os

import numpy as np
from skimage.draw import ellipse_perimeter, circle_perimeter
from tqdm import tqdm
import cv2

from detectron2.structures import BoxMode

CLASSES = {"label": 0, "button": 1}


def generate_bbox(px: List[int], py: List[int]) -> List[int]:
    """ generate smallest possible bounding box from segmentation mask
        x and y coords

    Args:
        px (List[int]): list of x coords
        py (List[int]): list of y coords

    Returns:
        List[int]: list of length 4 with location of
                   topleft and bottom right points of bounding box
    """
    return [np.min(px), np.min(py), np.max(px), np.max(py)]

def generate_gt_mask_coords(region: dict) -> Tuple[List[int], List[int]]:
    """ generate list of x coordinates (px) and y coordinates (py) for ground truth mask
    region format:
    {
        "shape_attributes": {
            "name": ...
            ...
            ...
        },
        "region_attributes": {
            "category_id": ...
            "pair": ...
        }
    }

    Args:
        region (dict): dictionary of information for one single mask in an image.
                       Format described above

    Returns:
        Tuple[List[int], List[int]]: list of x coordinates, list of y coordinates of perimeter of
                                     segmentation mask
    """
    s_attr = region["shape_attributes"]
    shape_type = s_attr["name"]

    if shape_type in ("polyline", "polygon"):
        px = s_attr["all_points_x"]
        py = s_attr["all_points_y"]
    elif shape_type == "ellipse":
        rr, cc = ellipse_perimeter(
            r=int(s_attr["cy"]),
            c=int(s_attr["cx"]),
            r_radius=int(s_attr["ry"]),
            c_radius=int(s_attr["rx"]),
            orientation=math.radians(s_attr["theta"]), # type: ignore
        )
    elif shape_type == "circle":
        rr, cc = circle_perimeter(r=int(s_attr["cy"]),
                                  c=int(s_attr["cx"]),
                                  radius=int(s_attr["r"]))
    elif shape_type == "rect":
        x, y = s_attr["x"], s_attr["y"]
        width, height = s_attr["width"], s_attr["height"]
        topleft = (x, y)
        topright = (x + width, y)
        bottomleft = (x, y + height)
        bottomright = (x + width, y + height)
        px = [topleft[0], bottomleft[0], bottomright[0], topright[0]]
        py = [topleft[1], bottomleft[1], bottomright[1], topright[1]]
    else:
        raise Exception("unsupported shape type")

    # sort px, py for ellipse and circle
    if shape_type in ("circle", "ellipse"):
        angle = np.arctan2(rr - np.mean(rr), cc - np.mean(cc))
        sorted_by_angle = np.argsort(angle)
        py = rr[sorted_by_angle]
        px = cc[sorted_by_angle]

    return px, py

def get_elevator_dicts(buildings: List[str]) -> List[dict]:
    """ Given a list of the building paths, return a
        list of Detectron2 standard dataset dicts

    Args:
        buildings (List[str]): List of paths to building folders

    Returns:
        List[dict]: List of Detectron2 dataset dictionaries
    """
    elevator_dicts = []
    image_id = 0
    for b in tqdm(buildings, position=0, desc="buildings"):
        annos_fp = glob(os.path.join(b, "*annotations.json"))[-1]
        with open(annos_fp) as f:
            img_dicts = json.load(f)

        for img_dict in tqdm(img_dicts.values(), position=1, desc="images", leave=False):
            record = {}
            im_path = os.path.join(b, img_dict["filename"])
            height, width = cv2.imread(im_path).shape[:2]

            record["file_name"] = im_path
            record["image_id"] = image_id
            record["height"] = height
            record["width"] = width

            gt_objs = []
            for r in img_dict["regions"]:
                px, py = generate_gt_mask_coords(region=r)
                poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
                poly = [p for x in poly for p in x]
                bbox = generate_bbox(px=px, py=py)

                gt_obj = {
                    "bbox": bbox,
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": [poly],
                    "category_id": CLASSES[r["region_attributes"]["category_id"]]
                }
                gt_objs.append(gt_obj)

            record["annotations"] = gt_objs
            elevator_dicts.append(record)

    return elevator_dicts