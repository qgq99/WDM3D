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
from torch.optim import Adam, AdamW, SGD
import time


FUSION_METHOD = {
    "plus": lambda x, y: x + y,
    "mul": lambda x, y: x * y,
    # "cat": lambda x, y: torch.cat([x, y], dim=1)    # 在通道维度cat
}

OPTIMIZER = {
    "Adam": Adam,
    "AdamW": AdamW,
    "SGD": SGD
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


def create_optimizer(model: torch.nn.Module, config: dict):
    return OPTIMIZER[config["module"]](params=model.parameters(), **config["params"])


def calc_model_params_count(model: torch.nn.Module):
    cnt = 0
    for p in model.parameters():
        if p.requires_grad == True:
            cnt += p.numel()
    return cnt / 1024 / 1024


def random_bbox2d(h=384, w=1280, cls_cnt=80, device="cpu"):
    """
    随机生成一个bbox标签, 形如[x1, y1, x2, y2, conf, cls]
    h, w: 假设标签属于一个尺寸为h*w的图像

    +10 -10 为防止生成的长度或宽度为0, 即x1 == x2 or y1 == y2
    """
    cls = random.randint(0, cls_cnt-1)
    conf = random.random()
    x1, y1 = random.random() * (h - 10), random.random() * (w - 10)

    x2 = random.random() * (h - x1 - 10) + x1 + 10
    y2 = random.random() * (w - y1 - 10) + y1 + 10
    return torch.tensor([x1, y1, x2, y2, conf, cls]).to(device)


def format_sec(s):
    """
    将秒数转换为易读格式
    :param s: 秒数
    :return:
    """
    D = 24 * 60 * 60  # 每天秒数
    H = 60 * 60  # 每小时秒数
    M = 60  # 每分钟秒数
    day = int(s // D)
    hour = int((s - day * D) // H)
    minute = int((s - day * D - hour * H) // M)
    seconds = s - day * D - hour * H - minute * M

    day = f"{day}天" if day > 0 else ""
    hour = f"{hour}小时" if hour > 0 else ""
    minute = f"{minute}分钟" if minute > 0 else ""
    return day + hour + minute + f"{seconds:.4f}秒"


class Timer:
    """
    用于计算并显示程序耗时的工具
    usage:
        with Timer("forward"):
            result = model(imgs)
    """

    def __init__(self, consumer="", work=True):
        """
        work: 是否工作
        """
        self.consumer = consumer
        self.work = work

    def __enter__(self):
        if self.work:
            self.start = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.work:
            print(
                f"Time consumption of [{self.consumer}]: {format_sec(time.time() - self.start)}")
        del self
