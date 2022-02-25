# nndet2

Object Detection models based on detectron2.

## Backbones
- [ConvNeXt](https://arxiv.org/abs/2201.03545) [config](configs/convnext)

| Model  | Backbone   | Lr schd | Mem (GB) | mask AP | box AP | Config | Download |
|:---------:|:-------:|:-------:|:--------:|:--------------:|:------:|:------:|:--------:|
| Mask RCNN      | ConvNeXt_Tiny | 1x      |       |      40.5      |  43.8  | [config](configs/convnext/mask_rcnn_convnext_tiny_fpn_1x.py) | [model]()|
| RetinaNet      | ConvNeXt_Tiny | 1x      |       |            |  43.8  | [config](configs/convnext/retinanet_convnext_tiny_fpn_1x.py) | [model]()|

- [ResNet](https://arxiv.org/abs/1512.03385) [config](configs/resnet)

| Model  | Backbone   | Lr schd | Mem (GB) | mask AP | box AP | Config | Download |
|:---------:|:-------:|:-------:|:--------:|:--------------:|:------:|:------:|:--------:|
| RetinaNet+      | ResNet50* | 1x      |       |            |  41.9  | [config](configs/resnet/retinanet_R_50_torchvision_FPN_1x.py ) | [model]()|

`+`: we use GN + GIoU
`*`: we use new TorchVision [SOTA ResNet models](https://github.com/pytorch/vision/issues/3995#issuecomment-1013906621)

