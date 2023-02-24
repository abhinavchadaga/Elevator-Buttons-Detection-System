import json
import os
from glob import glob
import shutil

from termcolor import cprint

base = "data/panels/ut_west_campus/"
buildings = [b for b in glob(os.path.join(base, "*")) if os.path.isdir(b)]
all_annos = {}
for b in buildings:
    bname = os.path.basename(b)
    extensions = [".jpg", ".JPG", ".JPEG", ".jpeg", ".png"]
    with open(os.path.join(b, "annotations.json")) as f:
        annos = json.load(f)

    for k, img_dict in annos.items():
        new_filename = f"{bname}_{k}"
        all_annos[new_filename] = img_dict
        all_annos[new_filename]["filename"] = new_filename
        src = os.path.join(base, bname, k)
        dest = os.path.join(base, new_filename)
        shutil.copy2(src=src, dst=dest)

    cprint(f"finished {bname}!!", "green")

with open("data/panels/ut_west_campus/annotations.json", "w") as f:
    json.dump(all_annos, f)

cprint("success!!", "green", attrs=["bold"])
