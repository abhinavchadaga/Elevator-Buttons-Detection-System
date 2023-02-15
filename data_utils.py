from asyncio.log import logger
import json
import logging
from typing import List, Tuple
import math
import os
from copy import deepcopy

import numpy as np
from tqdm import tqdm
from skimage.draw import ellipse_perimeter, circle_perimeter, polygon
import cv2

from detectron2.structures import BoxMode
from elevator_datasets import CLASSES


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


def generate_gt_mask_coords(
    region: dict, im_height: int, im_width: int
) -> Tuple[List[int], List[int]]:
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

    px = np.clip(px, 0, im_width - 1)
    py = np.clip(py, 0, im_height - 1)
    return px, py


def generate_gt_mask_opaque(
    region: dict, im_height: int, im_width: int
) -> Tuple[List[int], List[int]]:
    """Convert a perimeter mask used for Detectron2 into an opaque mask for


    Args:
        region (dict): same format as region for generate_gt_mask_coords

    Returns:
        Tuple[List[int], List[int]]: pixel coordinates of polygon. Can be used to index
            directly into an array i.e img[rr, cc] = 1
    """
    px, py = generate_gt_mask_coords(
        region=region, im_height=im_height, im_width=im_width
    )
    rr, cc = polygon(r=py, c=px)
    return rr, cc


def via_dict_to_d2_dict(
    img_dir: str, img_dict: dict, img_id: int, skip_no_pairs=False
) -> dict:
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

    annos = []
    for r in img_dict["regions"]:
        pair = r["region_attributes"].get("pair")
        if skip_no_pairs and (pair is None or pair == ""):
            continue
        px, py = generate_gt_mask_coords(region=r, im_height=height, im_width=width)
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


def read_split_file(fpath: str) -> List[List[str]]:
    """Read lines from a split file and partition images into three lists:
    train, val, test

    Args:
        fpath (str): path to the "split.txt" file

    Returns:
        List[List[str]]: train, val, test, list of images
    """
    out = [[], [], []]
    d = {"train": 0, "val": 1, "test": 2}
    with open(fpath) as f:
        for line in f:
            im_path, spl = line.rstrip().split(",")
            out[d[spl]].append(im_path)

    return out


def random_split_mixed_set(
    img_dir: str, split_ratio: Tuple[float, float, float], seed: int
) -> None:
    """Randomly split the data for the mixed elevators set.

    Args:
        img_dir (str): path to mixed elevators
        split_ratio (tuple): ratio to use for split.
        seed (int): for reproducibility.

    Returns:
        List[List[str]]: _description_
    """
    assert sum(split_ratio) == 1

    # open the annotations file
    with open(os.path.join(img_dir, "annotations.json")) as f:
        img_dicts: dict = json.load(f)

    im_paths = [os.path.join(img_dir, fname) for fname in img_dicts]
    # shuffle image paths randomly
    im_paths = np.array(im_paths)
    rng = np.random.default_rng(seed=seed)
    rng.shuffle(im_paths)
    trainset_size = int(len(im_paths) * split_ratio[0])
    valset_size = int(len(im_paths) * split_ratio[1])
    with open(os.path.join(img_dir, "split.txt"), "w") as f:
        for i in range(len(im_paths)):
            if i < trainset_size:
                spl = "train"
            elif i < trainset_size + valset_size:
                spl = "val"
            else:
                spl = "test"

            f.write(f"{im_paths[i]},{spl}\n")


def register_dataset(im_paths: List[str], skip_no_pairs=True) -> List[dict]:
    """from a list of paths to images, generate the dataset

    Args:
        im_paths (List[str]): list of image paths

    Returns:
        _type_: a list of Detectron2 dataset dicts
    """
    img_dir = os.path.dirname(im_paths[0])
    # open the annotations file
    with open(os.path.join(img_dir, "annotations.json")) as f:
        img_dicts: dict = json.load(f)

    dataset_dicts = []
    # add dicts to appropriate dataset
    for i in range(len(im_paths)):
        fname = os.path.basename(im_paths[i])
        img_dict = img_dicts[fname]
        d = via_dict_to_d2_dict(
            img_dir=img_dir, img_dict=img_dict, img_id=i, skip_no_pairs=skip_no_pairs
        )
        dataset_dicts.append(d)
    return dataset_dicts


def generate_missed_detections_data(dataset_name: str, skip_no_pairs=True) -> None:
    """Generate missed detections data from a pre-existing panels dataset

    Args:
        dataset_name (str): name of the panels dataset to use
        skip_no_pairs (bool, optional): ignore buttons with no pairs when
            constructing full class map. Defaults to True.
    """
    with open(os.path.join("data/panels/", dataset_name, "split.txt")) as f:
        im_paths = f.readlines()

    with open(os.path.join("data/panels", dataset_name, "annotations.json")) as f:
        annos = json.load(f)

    gts = {}
    spl = []
    save_dir = os.path.join("data/missed_detections", dataset_name)
    os.makedirs(save_dir, exist_ok=True)

    for l in tqdm(im_paths, desc="panel images"):
        # original image loop
        im_path, ds = l.split(",")
        filename = os.path.basename(im_path)
        img_dict = annos[filename]
        im = cv2.imread(im_path)
        height, width = im.shape[:2]
        all_regions = np.zeros_like(im, dtype=np.uint8)
        i = 0
        for r in img_dict["regions"]:
            # build class map w all detections
            pair = r["region_attributes"].get("pair")
            if skip_no_pairs and (pair is None or pair == ""):
                # skip features with no paired button or label
                continue
            rr, cc = generate_gt_mask_opaque(region=r, im_height=height, im_width=width)
            _class = r["region_attributes"]["category_id"]
            # green for label, blue for button
            all_regions[rr, cc] = [0, 255, 0] if _class == "label" else [0, 0, 255]

        # save none missing first
        new_fname = f"{os.path.splitext(filename)[0]}_none.jpg"
        gts[new_fname] = {
            "filename": new_fname,
            "original_image": filename,
            "regions": [],
        }
        save_path = os.path.join(save_dir, new_fname)
        cv2.imwrite(save_path, all_regions[:, :, ::-1])
        spl.append(f"{save_path},{ds}")

        for r in img_dict["regions"]:
            pair = r["region_attributes"].get("pair")
            if pair is None or pair == "":
                # No need to guess buttons w no pairs
                continue
            rr, cc = generate_gt_mask_opaque(region=r, im_height=height, im_width=width)
            minus_one = all_regions.copy()
            minus_one[rr, cc] = [0, 0, 0]

            new_fname = f"{os.path.splitext(filename)[0]}_{i}.jpg"
            gts[new_fname] = {
                "filename": new_fname,
                "original_image": filename,
                "regions": [deepcopy(r)],
            }

            save_path = os.path.join(save_dir, new_fname)
            cv2.imwrite(save_path, minus_one[:, :, ::-1])
            spl.append(f"{save_path},{ds}")
            i += 1

    with open(os.path.join(save_dir, "annotations.json"), "w") as f:
        json.dump(gts, f)

    with open(os.path.join(save_dir, "split.txt"), "w") as f:
        f.writelines(spl)
