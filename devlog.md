## 2024年11月10日
- 由于过度依赖mmcv且计算成本过高（使用了mmcv的MultiScaleDeformableAttention），暂计划放弃base_neck
- 若后续仍需要该模块，可从[Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/ops/modules/ms_deform_attn.py)获取MultiScaleDeformableAttention的纯torch实现


## 2024年11月11日

### Bug
- 报错:  one of the variables needed for gradient computation has been modified by an inplace operation: [torch.cuda.FloatTensor [8, 64, 96, 320]], which is output 0 of ReluBackward0, is at version 2; expected version 1 instead. Hint: enable anomaly detection to find the operation that failed to compute its gradient, with torch.autograd.set_detect_anomaly(True).
- 确定问题出在neck的forward, 在特征相加时尽量使用`a = a + b`的形式而非`a += b`, torch执行后者时是inplace操作, 如果其他地方对同一张量有若干inplace or non-inplace(例如nn.ReLu(inplace=Tarue)), 可能导致计算图确实节点进而导师反向传播时报错


## 2024年11月12日

### 调整生成深度图的程序
- 对点云可视化后发现，一张图片对应的点云是整个场景四周的所有点云，因此深度值有负数是正常的，表示点在向后房后，于是将深度值为负数的点也过滤掉
- 调整归一化方式，将深度度归一化到`[1,255]`, 像素值为0的表示该像素点缺失深度值, 同时记录深度值的实际最值， 待读取深度图时可将像素值还原为深度值
- 由于点云稀疏，因此一张图像的像素中只有一部分具有深度值，于是考虑通过插值提高深度信息的密度，该功能已经实现，但结果不够理想，插值得到的深度信息有一定失真，后续计划实验进一步验证

### 关于深度图生成的其他考虑
- 一些方法将天空的深度被认为是无穷大，于是考虑使用一个分割模型将天空分割出来，然后将归一化方法改为：深度值归一化到`[1, 254]`，0表示缺失深度，255表示深度无穷大，这样将像素点分为三类，在计算损失时可能会更好
- but，多数现成模型可分割的类别中并没有天空，若要单独训练一个，太过麻烦，另外引入一个分割模型会使得该程序耗时大幅提升，故暂未打算实施
    

## 2024年11月28日
### 关于loss计算
WeakM3D的loss计算过程用到了密度(density), 其是基于点云数据计算得到的，具体是通过计算每个点到其他所有点的距离，然后将一个点的密度定义为距离该点不超过某一阈值的点的数量，然而该过程复杂度较高(O(n2))，即使用了bbox select，一个实例的预测点云也有上万个点，这样的计算时间开销完全无法承受，因此WeakM3D对每个实例仅随机选取100个点作为该实例的objec-lidar, 并仅计算这100个点的密度。

然而100个点所包含的空间信息太少，理想状态应该是计算所有点，但现在没有找当更好的方法，只能也暂时采取该方法，应100设置为参数
