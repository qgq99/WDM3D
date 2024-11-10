## 2024年11月10日
- 由于过度依赖mmcv且计算成本过高（使用了mmcv的MultiScaleDeformableAttention），暂计划放弃base_neck
- 若后续仍需要该模块，可从[Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/ops/modules/ms_deform_attn.py)获取MultiScaleDeformableAttention的纯torch实现