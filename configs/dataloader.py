from omegaconf import OmegaConf
from detectron2.config import LazyCall as L
from detectron2.data import (
    get_detection_dataset_dicts,
    build_detection_train_loader,
    build_detection_test_loader,
    DatasetMapper,
)
import detectron2.data.transforms as T

dataloader = OmegaConf.create()

dataloader.image_size = 1024
dataloader.train = L(build_detection_train_loader)(
    dataset=L(get_detection_dataset_dicts)(
        names=["mixed_train", "ut_west_campus_train"]
    ),
    mapper=L(DatasetMapper)(
        is_train=True,
        augmentations=[
            L(T.ResizeScale)(
                min_scale=0.1,
                max_scale=2.0,
                target_height=dataloader.image_size,
                target_width=dataloader.image_size,
            ),
            L(T.FixedSizeCrop)(
                crop_size=(dataloader.image_size, dataloader.image_size), pad=False
            ),
        ],
        image_format="RGB",
        use_instance_mask=True,
        recompute_boxes=True,
    ),
    total_batch_size=4,
    num_workers=8,
)

dataloader.val = L(build_detection_test_loader)(
    dataset=L(get_detection_dataset_dicts)(
        names=["mixed_val", "ut_west_campus_val"], filter_empty=False
    ),
    mapper=L(DatasetMapper)(
        is_train=True,
        augmentations=[
            L(T.ResizeShortestEdge)(
                short_edge_length=dataloader.image_size,
                max_size=dataloader.image_size,
            ),
        ],
        image_format="BGR",
        use_instance_mask=True,
    ),
    num_workers=8,
)

dataloader.mixed_test = L(build_detection_test_loader)(
    dataset=L(get_detection_dataset_dicts)(names="mixed_test", filter_empty=False),
    mapper=L(DatasetMapper)(
        is_train=False,
        augmentations=[
            L(T.ResizeShortestEdge)(
                short_edge_length=dataloader.image_size,
                max_size=dataloader.image_size,
            ),
        ],
        image_format="BGR",
    ),
    num_workers=8,
)

dataloader.ut_west_campus_test = L(build_detection_test_loader)(
    dataset=L(get_detection_dataset_dicts)(
        names="ut_west_campus_test", filter_empty=False
    ),
    mapper=L(DatasetMapper)(
        is_train=False,
        augmentations=[
            L(T.ResizeShortestEdge)(
                short_edge_length=dataloader.image_size,
                max_size=dataloader.image_size,
            ),
        ],
        image_format="BGR",
    ),
    num_workers=8,
)
