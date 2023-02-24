from detectron2.data import (
    get_detection_dataset_dicts,
    build_detection_train_loader,
    build_detection_test_loader,
    DatasetMapper,
)

import detectron2.data.transforms as T


def build_coco_train_loader(cfg):
    return build_detection_train_loader(
        dataset=get_detection_dataset_dicts(cfg.DATASETS.TRAIN[0], filter_empty=False),
        mapper=DatasetMapper(
            is_train=True,
            augmentations=[
                T.ResizeScale(
                    min_scale=0.1,
                    max_scale=2.0,
                    target_height=1024,
                    target_width=1024,
                ),
                T.FixedSizeCrop(crop_size=(1024, 1024), pad=False),
            ],
            image_format="RGB",
            use_instance_mask=True,
            recompute_boxes=True,
        ),
        total_batch_size=cfg.SOLVER.IMS_PER_BATCH,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
    )


def build_coco_val_loader(cfg):
    return build_detection_test_loader(
        dataset=get_detection_dataset_dicts(cfg.DATASETS.VAL[0], filter_empty=False),
        mapper=DatasetMapper(
            is_train=True,
            augmentations=[T.ResizeShortestEdge(short_edge_length=1024, max_size=1024)],
            image_format="BGR",
            use_instance_mask=True,
        ),
    )
