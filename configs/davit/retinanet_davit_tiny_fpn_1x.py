from detectron2.config import LazyCall as L
from detectron2.solver.build import get_default_optimizer_params
from detectron2.modeling.backbone.fpn import LastLevelP6P7

from txdet.modeling.backbone import DaViT


model.backbone.bottom_up = L(DaViT)(
    embed_dims=(96, 192, 384, 768),
    depths=(1, 1, 3, 1), 
    num_heads=(3, 6, 12, 24),
    window_size=7,
    drop_path_rate=0.1,
    mlp_ratio=4.,
    overlapped_patch=False,
    ffn=True,
    cpe_act=False,
    out_features=["stage2", "stage3", "stage4"],
)
model.backbone.top_block=L(LastLevelP6P7)(
    in_channels=768,
    out_channels="${..out_channels}",
    in_feature="stage4"
)
model.pixel_mean = [123.675, 116.28, 103.53]
model.pixel_std = [58.395, 57.12, 57.375]
model.input_format = "RGB"

optimizer = L(torch.optim.AdamW)(
    params=L(get_default_optimizer_params)(
        weight_decay_norm=0.0
    ),  
    lr=1e-04,
    weight_decay=0.05,
    betas=(0.9, 0.999),
)

dataloader.train.mapper.image_format = "RGB"
dataloader.train.mapper.use_instance_mask = False 
