from ..common.coco_schedule import lr_multiplier_1x as lr_multiplier                                                 
from ..common.data.coco import dataloader
from ..common.models.retinanet import model
from ..common.train import train

import torch
from detectron2.config import LazyCall as L
from detectron2.solver.build import get_default_optimizer_params
from detectron2.modeling.backbone.fpn import LastLevelP6P7
from detectron2.modeling.backbone import FPN 

from nndet2.modeling.backbone import SwinTransformerV2


# replace backbone                                                                                                   
model.backbone = L(FPN)(
    bottom_up=L(SwinTransformerV2)(
        patch_size=4,
        in_chans=3,
        embed_dim=96,
        depths=[2, 2, 6, 2], 
        num_heads=[3, 6, 12, 24],
        window_size=16,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        pretrained_window_sizes=[8, 8, 8, 8],
        frozen_stages=-1,
        out_features=["stage2", "stage3", "stage4"],
        ),  
    in_features=["stage2", "stage3", "stage4"],
    out_channels=256,
    top_block=L(LastLevelP6P7)(in_channels=768, out_channels="${..out_channels}", in_feature="stage4")               
)
model.pixel_mean = [123.675, 116.28, 103.53]
model.pixel_std = [58.395, 57.12, 57.375]
model.input_format = "RGB"

optimizer = L(torch.optim.AdamW)(
    params=L(get_default_optimizer_params)(
        weight_decay_norm=0.0,
        overrides={
            "cpb_mlp": {"weight_decay": 0.0},
            "logit_scale": {"weight_decay": 0.0},
        }
    ),
    lr=1e-04,
    weight_decay=0.05,
    betas=(0.9, 0.999),
)

dataloader.train.mapper.image_format = "RGB"
dataloader.train.mapper.use_instance_mask = False  
