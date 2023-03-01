import os
from detectron2.modeling.backbone.fpn import LastLevelMaxPool

from torch import nn
from omegaconf import OmegaConf
from functools import partial
from fvcore.common.param_scheduler import MultiStepParamScheduler

from detectron2 import model_zoo
from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler
from detectron2.modeling.backbone.vit import (
    SimpleFeaturePyramid,
    ViT,
    get_vit_lr_decay_rate,
)

from .dataloader import dataloader

# start from mask rcnn with feature pyramid network
model = model_zoo.get_config("common/models/mask_rcnn_fpn.py").model
constants = model_zoo.get_config("common/data/constants.py").constants

# start from ImageNet pixel standardization
model.pixel_mean = constants.imagenet_rgb256_mean
model.pixel_std = constants.imagenet_rgb256_std
model.input_format = "RGB"

# replace backbone of model with simple feature pyramid from ViTDet
embed_dim, depth, num_heads, dp = 768, 12, 12, 0.1
model.backbone = L(SimpleFeaturePyramid)(
    net=L(ViT)(  # Single-scale ViT backbone
        img_size=1024,
        patch_size=16,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        drop_path_rate=dp,
        window_size=14,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        window_block_indexes=[
            # 2, 5, 8 11 for global attention
            0,
            1,
            3,
            4,
            6,
            7,
            9,
            10,
        ],
        residual_block_indexes=[],
        use_rel_pos=True,
        out_feature="last_feat",
    ),
    in_feature="${.net.out_feature}",
    out_channels=256,
    scale_factors=(4.0, 2.0, 1.0, 0.5),
    top_block=L(LastLevelMaxPool)(),
    norm="LN",
    square_pad=1024,
)

model.roi_heads.box_head.conv_norm = model.roi_heads.mask_head.conv_norm = "LN"

# 2conv in RPN:
model.proposal_generator.head.conv_dims = [-1, -1]

# 4conv1fc box head
model.roi_heads.box_head.conv_dims = [256, 256, 256, 256]
model.roi_heads.box_head.fc_dims = [1024]


# change number of roi heads to match classes in elevator dataset
model.roi_heads.num_classes = 2
model.roi_heads.box_predictor.test_score_thresh = 0.7

# configure training params
train = model_zoo.get_config("common/train.py").train
train.amp.enabled = True
train.ddp.fp16_compression = True
train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/mask_rcnn_vitdet_b/f325346929/model_final_61ccd1.pkl"
train.max_iter = 3350  # 50 epochs
train.eval_period = 134  # every 2 epochs
train.output_dir = "models/segmentation_vit"
train.finetuned_weights = os.path.join(train.output_dir, "model_best.pth")

# lr multiplier
lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1, 0.01],
        milestones=[int(0.8 * train.max_iter), int(0.9 * train.max_iter)],
        num_updates=train.max_iter,
    ),
    warmup_length=250 / train.max_iter,
    warmup_factor=0.001,
)

# optimizer
optimizer = model_zoo.get_config("common/optim.py").AdamW
optimizer.params.base_lr = 3e-5
optimizer.lr = 3e-5
optimizer.params.lr_factor_func = partial(
    get_vit_lr_decay_rate, num_layers=12, lr_decay_rate=0.7
)
optimizer.params.overrides = {"pos_embed": {"weight_decay": 0.0}}
