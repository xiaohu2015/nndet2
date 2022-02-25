from ..common.coco_schedule import lr_multiplier_1x as lr_multiplier
from ..common.data.coco import dataloader
from ..common.models.retinanet import model
from ..common.train import train

import torch
from detectron2.config import LazyCall as L
from detectron2.solver.build import get_default_optimizer_params
from detectron2.modeling.backbone import BasicStem, FPN, ResNet

dataloader.train.mapper.use_instance_mask = False
dataloader.train.mapper.image_format = "RGB"

model.backbone.bottom_up=L(ResNet)(
    stem=L(BasicStem)(in_channels=3, out_channels=64, norm="FrozenBN"),
    stages=L(ResNet.make_default_stages)(
        depth=50,
        stride_in_1x1=False,
        norm="FrozenBN",
    ),
    out_features=["res3", "res4", "res5"],
  freeze_at=0,
)
# GN head
model.head.norm = 'GN'
# giou loss
model.box_reg_loss_type = 'giou'
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
