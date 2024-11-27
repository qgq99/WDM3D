"""
-*- coding: utf-8 -*-
@Time    : 2024-11-13 12:30:49
@Author  : Qin Guoqing
@File    : wdm3d_utils.py
@Description : Description of this file
"""
import yaml
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random


FUSION_METHOD = {
    "plus": lambda x, y: x + y,
    "mul": lambda x, y: x * y,
    # "cat": lambda x, y: torch.cat([x, y], dim=1)    # 在通道维度cat
}


def create_module(g: dict, cfg: dict, module_name: str):
    """
    g: 应为globals()的返回值
    cfg: 配置对象
    """

    return g[cfg[module_name]["module"]](**cfg[module_name]["params"])


def load_config(config_path: str = "/home/qinguoqing/project/WDM3D/config/exp/exp.yaml", sub_cfg_keys=["dataset", "model"]):
    """
    加载配置对象, 包括数据配置、模型配置、损失配置等
    config_path:
    sub_cfg_keys: 需要加载的子配置文件
    """
    config = None
    with open(config_path) as f:
        config = yaml.safe_load(f)

    for k in sub_cfg_keys:
        config[k] = load_config(config[k], sub_cfg_keys=[])[k]

    return config


def kitti_collate(batch):
    img, target, original_idx = zip(*batch)
    img = torch.stack(list(img))
    return img, target, original_idx


def create_dataloader(dataset: Dataset, batch_size=8, shuffle=False, num_workers=4, collate_fn=kitti_collate):
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                            shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)

    return dataloader


def calc_model_params_count(model: torch.nn.Module):
    cnt = 1
    for p in model.parameters():
        if p.requires_grad == True:
            cnt += p.numel()
    return cnt / 1024 / 1024



def calc_batch_lidar_y_center(point_cloud, instance_cnt):
    """
    instance_cnt: bbox的数量, 应等于len(point_cloud)
    """
    batch_lidar_y_center = np.zeros((instance_cnt, 1), dtype=np.float32)

    for i in range(instance_cnt):
        batch_lidar_y_center[i] = np.mean(point_cloud[i][:, 1])
    

    return batch_lidar_y_center



def random_bbox2d(h=384, w=1280, cls_cnt=80, device="cpu"):
    """
    随机生成一个bbox标签, 形如[x1, y1, x2, y2, conf, cls]
    h, w: 假设标签属于一个尺寸为h*w的图像
    """
    cls = random.randint(0, cls_cnt-1)
    conf = random.random()
    x1, y1 = random.random() * h, random.random() * w

    while True:
        x2, y2 = random.random() * h, random.random() * w
        if x2 >= x1 and y2 >= y1:
            break
    
    return torch.tensor([x1, y1, x2, y2, conf, cls]).to(device)