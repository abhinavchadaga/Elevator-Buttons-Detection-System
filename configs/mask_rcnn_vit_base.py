import os

from omegaconf import OmegaConf
from functools import partial
from fvcore.common.param_scheduler import MultiStepParamScheduler

from detectron2 import model_zoo
from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler
from detectron2.modeling.backbone.vit import get_vit_lr_decay_rate

from .dataloader import dataloader

# cfg = OmegaConf.create()

# get base model config
model = model_zoo.get_config("common/models/mask_rcnn_vitdet.py").model
model.roi_heads.num_classes = 2
model.roi_heads.box_predictor.test_score_thresh = 0.7
model.roi_heads.box_predictor.test_topk_per_image = 200

# configure training params
train = model_zoo.get_config("common/train.py").train
train.amp.enabled = True
train.ddp.fp16_compression = True
train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/mask_rcnn_vitdet_b/f325346929/model_final_61ccd1.pkl"
train.max_iter = 3350  # 50 epochs
train.eval_period = 134  # every 2 epochs
train.output_dir = "models/segmentation_v1"
train.finetuned_weights = os.path.join(train.output_dir, "model_best.pth")
# train.finetuned_weights = "models/mrcnn_vit_b/model_best.pth"

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
