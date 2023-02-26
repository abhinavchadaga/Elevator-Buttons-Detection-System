from functools import cmp_to_key
from itertools import repeat
import json
import random
from typing import List, Tuple
import math
import os
from copy import deepcopy
from collections import defaultdict

from multiprocessing import Pool
import numpy as np
from termcolor import cprint
from tqdm import tqdm
from skimage.draw import ellipse_perimeter, circle_perimeter, polygon
import cv2
from PIL import Image, ImageOps

from detectron2.structures import BoxMode

CLASSES = {"label": 0, "button": 1}


def generate_bbox(
    px: List[int], py: List[int], im_height: int, im_width: int
) -> List[int]:
    """Generate smallest possible bounding box from segmentation mask x and y coords.

    Args:
        px (List[int]): list of x coords
        py (List[int]): list of y coords

    Returns:
        List[int]: list of length 4 with location of topleft and bottom right points of
            bounding box
    """
    x1, y1, x2, y2 = np.min(px), np.min(py), np.max(px), np.max(py)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(im_width - 1, x2), min(im_height - 1, y2)
    return x1, y1, x2, y2


def generate_gt_mask_coords(
    region: dict, im_height: int, im_width: int
) -> Tuple[List[int], List[int]]:
    """
    generate list of x coordinates (px) and y coordinates (py) for ground truth mask
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
            orientation=s_attr["theta"],
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
        pair = pair.rstrip()
        px, py = generate_gt_mask_coords(region=r, im_height=height, im_width=width)
        poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
        poly = [p for x in poly for p in x]
        bbox = generate_bbox(px=px, py=py, im_height=height, im_width=width)

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


def random_split_ut_west_campus_set(
    img_dir: str, split_ratio: Tuple[float, float, float], seed: int
):
    """
    Randomly split the ut_west_campus dataset ensuring that images of the same
    building are placed in the same split

    Args:
        img_dir (str): path to images
        split_ratio (Tuple[float, float, float]): ratio to split BUILDINGS
        seed (int): for shuffle reproducibility
    """
    with open("data/panels/ut_west_campus/buildings.txt") as f:
        buildings = np.array([l.rstrip() for l in f])

    rng = np.random.default_rng(seed=seed)
    rng.shuffle(buildings)
    trainset_size = int(len(buildings) * split_ratio[0])
    valset_size = int(len(buildings) * split_ratio[1])
    split = {}
    for i, b in enumerate(buildings):
        if i < trainset_size:
            split[b] = "train"
        elif i < trainset_size + valset_size:
            split[b] = "val"
        else:
            split[b] = "test"

    with open("data/panels/ut_west_campus/annotations.json") as f:
        img_dicts = json.load(f)

    im_paths = [os.path.join(img_dir, fname) for fname in img_dicts]
    with open("data/panels/ut_west_campus/split.txt", "w") as f:
        for im_path in im_paths:
            bname = os.path.basename(im_path).split("_")[:-1]
            bname = "_".join(bname)
            f.write(f"{im_path},{split[bname]}\n")


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
    for i in tqdm(
        range(len(im_paths)), desc=f"registering {os.path.basename(img_dir)} dataset"
    ):
        fname = os.path.basename(im_paths[i])
        img_dict = img_dicts[fname]
        d = via_dict_to_d2_dict(
            img_dir=img_dir, img_dict=img_dict, img_id=i, skip_no_pairs=skip_no_pairs
        )
        dataset_dicts.append(d)
    return dataset_dicts


### MISSED DETECTIONS DATA GENERATION


def per_image(
    line: str, img_dict: dict, save_dir: str, skip_no_pairs: bool, verify: bool
):
    im_path, ds = line.split(",")
    filename = os.path.basename(im_path)
    im = cv2.imread(im_path)
    height, width = im.shape[:2]

    gts = {}
    spl = []

    # base class map
    id = 0
    base = np.zeros_like(im, dtype=np.uint8)
    pairs = defaultdict(lambda: [None] * 2)
    for r in img_dict["regions"]:
        pair = r["region_attributes"].get("pair")
        if skip_no_pairs and (pair is None or pair == ""):
            # skip features with no paired button or label
            continue

        pair = pair.rstrip()

        rr, cc = generate_gt_mask_opaque(region=r, im_height=height, im_width=width)
        _class = r["region_attributes"]["category_id"]
        if _class == "label":
            pairs[pair][0] = r
        else:
            pairs[pair][1] = r
        # green for label, blue for button
        base[rr, cc] = [0, 255, 0] if _class == "label" else [0, 0, 255]

    # save base/none missing first
    new_fname = f"{os.path.splitext(filename)[0]}_none.jpg"
    gts[new_fname] = {
        "filename": new_fname,
        "original_image": filename,
        "regions": [],
    }
    save_path = os.path.join(save_dir, new_fname)
    cv2.imwrite(save_path, base[:, :, ::-1])
    spl.append(f"{save_path},{ds}")

    def helper(k, id, remove="all"):
        """generate permutations of k buttons and labels removed from an image

        Args:
            k (_type_): number of buttons or labels to remove from an image
        """
        pool = deepcopy(pairs)  # button label pairs to choose from
        chosen = {}  # keep track of regions chosen so far
        while pool:
            out = base.copy()
            if k <= len(pool):
                selected_keys = random.sample(list(pool.keys()), k)
            else:
                # less than k pairs in the pool
                # resample already chosen keys and add remaining keys in pool
                selected_keys = random.sample(list(chosen.keys()), k - len(pool))
                selected_keys.extend(list(pool.keys()))

            regions = []
            for s in selected_keys:
                if s in pool:
                    # remove from pool, add to chosen
                    pair = pool.pop(s)
                    chosen[s] = pair
                else:
                    # s is not in pool, resampling from chosen
                    pair = chosen[s]

                label, button = pair
                if label is None or button is None:
                    continue
                if remove == "all":
                    r = label if random.random() < 0.5 else button
                elif remove == "label":
                    r = label
                elif remove == "button":
                    r = button
                else:
                    raise Exception("remove must be one of all, label or button")

                regions.append(r)
                rr, cc = generate_gt_mask_opaque(
                    region=r, im_height=height, im_width=width
                )
                out[rr, cc] = [0, 0, 0]

            # kinda expensive
            if verify:
                print("here!!")
                unique = set()
                for r in regions:
                    if r["region_attributes"]["pair"] in unique:
                        print([r["region_attributes"]["pair"] for r in regions])
                        raise Exception("non unique pairs!!!")
                    else:
                        unique.add(r["region_attributes"]["pair"])

            new_fname = f"{os.path.splitext(filename)[0]}_{id}.jpg"
            gts[new_fname] = {
                "filename": new_fname,
                "original_image": filename,
                "regions": regions,
            }

            save_path = os.path.join(save_dir, new_fname)
            cv2.imwrite(save_path, out[:, :, ::-1])
            spl.append(f"{save_path},{ds}")
            id += 1

        return id

    for k in range(1, max(2, math.ceil(0.2 * len(pairs) * 2))):
        try:
            if k == 1:
                id = helper(1, id, remove="button")
                id = helper(1, id, remove="label")
            else:
                id = helper(k, id, remove="all")
        except:
            cprint(f"error!!! {filename}", "red")

    cprint(f"successfully finished {filename}!!!", "green")
    return gts, spl


def generate_missed_detections_data(
    dataset_name: str, skip_no_pairs=True, verify=False
) -> None:
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

    all_gts, all_spl = {}, []
    save_dir = os.path.join("data/missed_detections", dataset_name)
    os.makedirs(save_dir, exist_ok=True)

    pool = Pool(processes=32)

    # gather args
    img_dicts = [annos[os.path.basename(l.split(",")[0])] for l in im_paths]
    args = zip(
        im_paths, img_dicts, repeat(save_dir), repeat(skip_no_pairs), repeat(verify)
    )
    results = pool.starmap(per_image, args)
    pool.close()
    pool.join()

    for x in results:
        all_gts.update(x[0])
        all_spl.extend(x[1])

    cprint("dumping data", "blue")
    with open(os.path.join(save_dir, "annotations.json"), "w") as f:
        json.dump(all_gts, f)

    with open(os.path.join(save_dir, "split.txt"), "w") as f:
        f.writelines(all_spl)

    cprint("success!!", "green", attrs=["bold"])


######## GENERATE CROPPED LABEL IMAGES ########


def generate_label_imgs(dataset_name: str, save_height: int, save_width: int):
    """
    generate label images and contain in an image size of save_height x save_width

    Args:
        dataset_name (str): _description_
        save_height (int): _description_
        save_width (int): _description_
    """
    assert dataset_name in ("mixed", "ut_west_campus")
    split_file_path = f"data/panels/{dataset_name}/split.txt"
    annos_path = f"data/panels/{dataset_name}/annotations.json"

    with open(split_file_path) as f:
        lines = [line.rstrip() for line in f]

    with open(annos_path) as f:
        annos = json.load(f)

    gtfiles = {"train": [], "val": [], "test": []}
    save_dir = os.path.join("data/labels")
    for line in tqdm(lines, desc="images"):
        im_path, split = line.split(",")
        filename = os.path.basename(im_path)
        img_dict = annos[filename]
        img = cv2.imread(im_path)
        height, width = img.shape[:2]
        for r in img_dict["regions"]:
            category_id = r["region_attributes"].get("category_id")

            ## ERROR CHECKING ##
            if category_id is None:
                print("missing category id: ", filename)
                continue
            ####################

            if category_id != "label":
                continue

            gt = r["region_attributes"].get("pair")

            ## ERROR CHECKING ##
            if gt is None:
                print("missing pair attribute: ", filename)
                continue
            ####################

            gt = gt.rstrip().replace(" ", "_")

            # create binary mask of label
            binary_mask = np.zeros_like(img, dtype=np.uint8)
            rr, cc = generate_gt_mask_opaque(r, height, width)
            binary_mask[rr, cc, :] = 255

            # only keep pixels that correspond to label
            label_img = cv2.bitwise_and(img, binary_mask)

            # use bbox to crop image
            bbox = generate_bbox(px=cc, py=rr, im_height=height, im_width=width)
            x1, y1, x2, y2 = bbox
            label_img = label_img[y1:y2, x1:x2]

            # resize img to fixed height and width
            label_img = Image.fromarray(label_img[:, :, ::-1])
            resized_img = ImageOps.pad(label_img, (save_width, save_height))

            # save image into appropriate split
            save_path = os.path.join(save_dir, split)
            save_filename = f"{os.path.splitext(img_dict['filename'])[0]}-{gt}.jpg"
            os.makedirs(save_path, exist_ok=True)
            save_path = os.path.join(save_path, save_filename)
            resized_img.save(save_path)

            # write to gt file
            gtfiles[split].append(f"{save_filename} {gt}\n")

    # write gt files
    for k, v in gtfiles.items():
        with open(os.path.join("data/labels", k, "gt.txt"), "a") as f:
            f.writelines(v)


## LABEL BUTTON ASSOCIATION DATA GENERATION


def per_image_2(line, img_dict, save_dir):
    im_path, split = line.split(",")
    filename = os.path.basename(im_path)
    img = cv2.imread(im_path)
    height, width = img.shape[:2]

    # GENERATE PAIRS DICT
    # pair -> label, button, pair_x_center, pair_y_center
    pairs = defaultdict(lambda: {"label": None, "button": None})
    for r in img_dict["regions"]:
        pair = r["region_attributes"].get("pair")
        if pair is None or pair == "":
            # skip features with no paired button or label
            continue

        pair = pair.rstrip()

        category_id = r["region_attributes"]["category_id"]
        # generate mask
        mask = generate_gt_mask_opaque(r, height, width)
        # calculate center
        bbox = generate_bbox(mask[1], mask[0], height, width)
        center = (bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2
        # update pairs dict
        pairs[pair][category_id] = {"mask": mask, "center": center}

    #### ERROR CHECKING!!!! ###############
    skip = False
    for k, v in pairs.items():
        if v["label"] is None or v["button"] is None:
            print(filename, " ", k)
            skip = True

    if skip:
        return [None]
    #######################################

    ret = []
    # GENERATE TRAINING DATA FOR EACH LABEL
    for pair, tgt in pairs.items():
        out = np.zeros_like(img, dtype=np.uint8)
        label = tgt["label"]

        # mark label to associate with as red
        out[label["mask"]] = [255, 0, 0]

        def compare(a, b):
            return math.dist(label["center"], a["button"]["center"]) - math.dist(
                label["center"], b["button"]["center"]
            )

        nearest_4 = sorted(list(pairs.values()), key=cmp_to_key(compare))[:4]

        # assign gt index
        gt = None
        for i, p in enumerate(nearest_4):
            if p["label"]["center"] == label["center"]:
                # tgt pair
                gt = i
            else:
                # non-tgt pair
                out[p["label"]["mask"]] = [0, 255, 0]  # non-tgt labels are green

            out[p["button"]["mask"]] = [0, 0, 255]  # buttons are blue

        assert gt is not None
        pair = pair.replace(" ", "_")
        save_path = os.path.join(
            save_dir, f"{os.path.splitext(filename)[0]}-{pair}-{gt}.jpg"
        )
        cv2.imwrite(save_path, out[:, :, ::-1])
        ret.append(f"{save_path},{split}\n")

    cprint(f"finished {filename}!!!", "green")
    return ret


def generate_button_label_association_imgs(dataset_name: str) -> None:
    assert dataset_name in ("mixed", "ut_west_campus")
    split_file_path = f"data/panels/{dataset_name}/split.txt"
    annos_path = f"data/panels/{dataset_name}/annotations.json"

    with open(split_file_path) as f:
        lines = [line.rstrip() for line in f]

    with open(annos_path) as f:
        annos = json.load(f)

    save_dir = f"data/btn_label_assoc/{dataset_name}"
    os.makedirs(save_dir, exist_ok=True)
    pool = Pool(processes=32)

    # gather args
    img_dicts = [annos[os.path.basename(l.split(",")[0])] for l in lines]
    args = zip(lines, img_dicts, repeat(save_dir))
    result = pool.starmap(per_image_2, args)

    with open(os.path.join(save_dir, "split.txt"), "w") as f:
        for l in result:
            if l[0] is not None:
                f.writelines(l)

    cprint("done!!", "green", attrs=["bold"])


if __name__ == "__main__":
    generate_button_label_association_imgs("ut_west_campus")
    # generate_label_imgs("ut_west_campus", 32, 128)
    # generate_missed_detections_data("ut_west_campus", True)
