import json
from typing import List, Tuple
import math
import os

import numpy as np
from tqdm import tqdm
from skimage.draw import ellipse_perimeter, circle_perimeter
import cv2

from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog

CLASSES = {"label": 0, "button": 1}


def generate_bbox(px: List[int], py: List[int]) -> List[int]:
    """Generate smallest possible bounding box from segmentation mask x and y coords.

    Args:
        px (List[int]): list of x coords
        py (List[int]): list of y coords

    Returns:
        List[int]: list of length 4 with location of topleft and bottom right points of
                   bounding box
    """
    return [np.min(px), np.min(py), np.max(px), np.max(py)]


def generate_gt_mask_coords(region: dict) -> Tuple[List[int], List[int]]:
    """generate list of x coordinates (px) and y coordinates (py) for ground truth mask
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
        Tuple[List[int], List[int]]: list of x coordinates, list of y coordinates of
                                     perimeter of segmentation mask
    """
    s_attr = region["shape_attributes"]
    shape_type = s_attr["name"]

    rr, cc = None, None

    px, py = [], []
    if shape_type in ("polyline", "polygon"):
        px: List[int] = s_attr["all_points_x"]
        py: List[int] = s_attr["all_points_y"]
    elif shape_type == "ellipse":
        rr, cc = ellipse_perimeter(
            r=int(s_attr["cy"]),
            c=int(s_attr["cx"]),
            r_radius=int(s_attr["ry"]),
            c_radius=int(s_attr["rx"]),
            orientation=math.radians(s_attr["theta"]),  # type: ignore
        )
    elif shape_type == "circle":
        rr, cc = circle_perimeter(
            r=int(s_attr["cy"]), c=int(s_attr["cx"]), radius=int(s_attr["r"])
        )
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
        assert rr is not None and cc is not None
        angle = np.arctan2(rr - np.mean(rr), cc - np.mean(cc))
        sorted_by_angle = np.argsort(angle)
        py = rr[sorted_by_angle]
        px = cc[sorted_by_angle]

    return px, py


def elevator_dict_to_d2_dict(img_dir: str, img_dict: dict, img_id: int) -> dict:
    """Convert an image dict from VGG annotation file into a single standard Detectron2
    dataset dict.

        this function should be called in a loop where img_id is incremented so that
        each d2 dict is assigned a unique id

    Args:
        img_dir (str): path to directory of img
        img_dict (dict): dictionary containing following keys
                        - filename
                        - size,
                        - regions
                        - file attributes
        img_id (int): unique id to identify image

    Returns:
        dict: _description_
    """
    d = dict()
    im_path = os.path.join(img_dir, img_dict["filename"])
    height, width = cv2.imread(im_path).shape[:2]

    d["file_name"] = im_path
    d["image_id"] = img_id
    d["height"] = height
    d["width"] = width

    annos = list()
    for r in img_dict["regions"]:
        px, py = generate_gt_mask_coords(region=r)
        poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
        poly = [p for x in poly for p in x]
        bbox = generate_bbox(px=px, py=py)

        anno = {
            "bbox": bbox,
            "bbox_mode": BoxMode.XYXY_ABS,
            "segmentation": [poly],
            "category_id": CLASSES[r["region_attributes"]["category_id"]],
        }
        annos.append(anno)

    d["annotations"] = annos
    return d


def random_split_mixed_set(
    img_dir: str, split_ratio=(0.7, 0.1, 0.2), seed=10
) -> List[List[str]]:
    """Randomly split the data for the mixed elevators set.

    Args:
        img_dir (str): path to mixed elevators
        split_ratio (tuple, optional): ratio to use for split. Defaults to
                                       (0.7, 0.1, 0.2).
        seed (int, optional): for reproducibility. Defaults to 10.

    Returns:
        List[List[str]]: _description_
    """
    assert sum(split_ratio) == 1

    # open the annotations file
    with open(os.path.join(img_dir, "mixed_annotations.json")) as f:
        img_dicts: dict = json.load(f)

    im_paths = [os.path.join(img_dir, fname) for fname in img_dicts]
    # shuffle image paths randomly
    im_paths = np.array(im_paths)
    rng = np.random.default_rng(seed=seed)
    rng.shuffle(im_paths)
    trainset_size = int(len(im_paths) * split_ratio[0])
    valset_size = int(len(im_paths) * split_ratio[1])
    datasets = np.array_split(im_paths, [trainset_size, trainset_size + valset_size])
    datasets = [list(d) for d in datasets]
    return datasets


def get_mixed_set_dicts(im_paths: List[str]) -> List[dict]:
    """from a list of paths to images, generate the dataset

    Args:
        im_paths (List[str]): list of image paths

    Returns:
        _type_: a list of Detectron2 dataset dicts
    """
    img_dir = os.path.dirname(im_paths[0])
    # open the annotations file
    with open(os.path.join(img_dir, "mixed_annotations.json")) as f:
        img_dicts: dict = json.load(f)

    dataset_dicts = []
    # add dicts to appropriate dataset
    for i in tqdm(range(len(im_paths)), desc="loading images"):
        fname = im_paths[i].split("/")[-1]
        img_dict = img_dicts[fname]
        d = elevator_dict_to_d2_dict(img_dir=img_dir, img_dict=img_dict, img_id=i)
        dataset_dicts.append(d)
    return dataset_dicts


# def get_elevator_dicts(buildings: List[str]) -> List[dict]:
#     """ Given a list of the building paths, return a
#         list of Detectron2 standard dataset dicts

#     Args:
#         buildings (List[str]): List of paths to building folders

#     Returns:
#         List[dict]: List of Detectron2 dataset dictionaries
#     """
#     elevator_dicts = []
#     image_id = 0
#     for b in tqdm(buildings, position=0, desc="buildings"):
#         annos_fp = glob(os.path.join(b, "*annotations.json"))[-1]
#         with open(annos_fp) as f:
#             img_dicts = json.load(f)

#         for img_dict in tqdm(img_dicts.values(), position=1, desc="images", leave=False):
#             record = {}
#             im_path = os.path.join(b, img_dict["filename"])
#             height, width = cv2.imread(im_path).shape[:2]

#             record["file_name"] = im_path
#             record["image_id"] = image_id
#             record["height"] = height
#             record["width"] = width

#             gt_objs = []
#             for r in img_dict["regions"]:
#                 px, py = generate_gt_mask_coords(region=r)
#                 poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
#                 poly = [p for x in poly for p in x]
#                 bbox = generate_bbox(px=px, py=py)

#                 gt_obj = {
#                     "bbox": bbox,
#                     "bbox_mode": BoxMode.XYXY_ABS,
#                     "segmentation": [poly],
#                     "category_id": CLASSES[r["region_attributes"]["category_id"]]
#                 }
#                 gt_objs.append(gt_obj)

#             record["annotations"] = gt_objs
#             elevator_dicts.append(record)

#     return elevator_dicts


if __name__ == "__main__":
    img_paths = random_split_mixed_set("data/panels/mixed")
    for name, paths in zip(["train", "val", "test"], img_paths):
        DatasetCatalog.register(
            name=f"mixed_{name}",
            func=lambda im_paths=paths: get_mixed_set_dicts(im_paths=im_paths),
        )

    trainset: List[dict] = DatasetCatalog.get("mixed_train")
    print(len(trainset))
