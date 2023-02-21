from omegaconf import OmegaConf

import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detectron2.data import (
    DatasetMapper,
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts,
)
from detectron2.evaluation import COCOEvaluator

dataloader = OmegaConf.create()


image_size = 1024  # all images cropped to 1024 by 1024
dataloader.train = L(build_detection_train_loader)(
    dataset=L(get_detection_dataset_dicts)(names="mixed_train", filter_empty=False),
    mapper=L(DatasetMapper)(
        is_train=True,
        augmentations=[
            T.ResizeScale(
                min_scale=0.1,
                max_scale=2.0,
                target_height=image_size,
                target_width=image_size,
            ),
            T.FixedSizeCrop(crop_size=(image_size, image_size), pad=False),
        ],
        image_format="BGR",
        use_instance_mask=True,
        recompute_boxes=True,
    ),
    total_batch_size=4,
    num_workers=8,
)

dataloader.val = L(build_detection_test_loader)(
    dataset=L(get_detection_dataset_dicts)(names="mixed_val", filter_empty=False),
    mapper=L(DatasetMapper)(
        is_train=True,
        augmentations=[
            L(T.ResizeShortestEdge)(short_edge_length=image_size, max_size=image_size),
        ],
        image_format="BGR",
        use_instance_mask=True,
    ),
    num_workers=8,
)

dataloader.test = L(build_detection_test_loader)(
    dataset=L(get_detection_dataset_dicts)(names="mixed_test", filter_empty=False),
    mapper=L(DatasetMapper)(
        is_train=False,
        augmentations=[
            L(T.ResizeShortestEdge)(short_edge_length=image_size, max_size=image_size),
        ],
        image_format="BGR",
        use_instance_mask=True,
    ),
    num_workers=8,
)

dataloader.evaluator = L(COCOEvaluator)(
    dataset_name="${..test.dataset.names}",
    use_fast_impl=False,
    allow_cached_coco=False,
    output_dir="eval_results",
)
