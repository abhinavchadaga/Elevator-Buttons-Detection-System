import os

from termcolor import cprint

cprint("converting training set...", "green")
os.system(
    "python3 parseq/tools/create_lmdb_dataset.py data/labels/train data/labels/train/gt.txt parseq/data/train/real True"
)
cprint("\nconverting validation set...", "green")
os.system(
    "python3 parseq/tools/create_lmdb_dataset.py data/labels/val data/labels/val/gt.txt parseq/data/val True"
)
cprint("\nconverting test set...", "green")
os.system(
    "python3 parseq/tools/create_lmdb_dataset.py data/labels/test data/labels/test/gt.txt parseq/data/test True"
)
cprint("\ndone!!", "green", attrs=["bold"])
