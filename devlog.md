## 2024年11月10日
- 由于过度依赖mmcv且计算成本过高（使用了mmcv的MultiScaleDeformableAttention），暂计划放弃base_neck
- 若后续仍需要该模块，可从[Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/ops/modules/ms_deform_attn.py)获取MultiScaleDeformableAttention的纯torch实现


## 2024年11月11日

### Bug
- 报错:  one of the variables needed for gradient computation has been modified by an inplace operation: [torch.cuda.FloatTensor [8, 64, 96, 320]], which is output 0 of ReluBackward0, is at version 2; expected version 1 instead. Hint: enable anomaly detection to find the operation that failed to compute its gradient, with torch.autograd.set_detect_anomaly(True).
- 确定问题出在neck的forward, 在特征相加时尽量使用`a = a + b`的形式而非`a += b`, torch执行后者时是inplace操作, 如果其他地方对同一张量有若干inplace or non-inplace(例如nn.ReLu(inplace=Tarue)), 可能导致计算图确实节点进而导师反向传播时报错