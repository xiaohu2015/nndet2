from ..common.coco_schedule import lr_multiplier_3x as lr_multiplier
from ..common.data.coco import dataloader
from ..common.models.mask_rcnn_fpn import model
from ..common.train import train

import torch
from detectron2.config import LazyCall as L
from detectron2.solver.build import get_default_optimizer_params

from nndet2.modeling.backbone import ConvNeXt


model.backbone.bottom_up = L(ConvNeXt)(
    in_chans=3,
    depths=[3, 3, 9, 3],
    dims=[96, 192, 384, 768], 
    drop_path_rate=0.4,
    layer_scale_init_value=1.0,
    out_features=["stage1", "stage2", "stage3", "stage4"],
)
model.pixel_mean = [123.675, 116.28, 103.53]
model.pixel_std = [58.395, 57.12, 57.375]
model.input_format = "RGB"

optimizer = L(torch.optim.AdamW)(
    params=L(get_default_optimizer_params)(
        # params.model is meant to be set to the model object, before instantiating
        # the optimizer.
        weight_decay_norm=0.0
    ),
    lr=1e-04,
    weight_decay=0.05,
    betas=(0.9, 0.999),
)

dataloader.train.mapper.image_format = "RGB"
