"""
-*- coding: utf-8 -*-
@Time    : 2024-11-09 20:46:07
@Author  : Qin Guoqing
@File    : temp.py
@Description : Description of this file
"""

from utils.wdm3d_utils import load_config, create_module, Timer
import torch
from model.backbone.fastvit.models import fastvit_t12
import os

G = globals()


def main():
    device = torch.device("cuda:0")
    # cfg = load_config("/home/qinguoqing/project/WDM3D/config/yolo/yolov9-s.yaml", sub_cfg_keys=[])
    fastvit = fastvit_t12(fork_feat=True)
    checkpoint = torch.load(
        "/home/qinguoqing/project/WDM3D/weight/fastvit_t12.pth.tar", weights_only=True)
    fastvit.load_state_dict(checkpoint['state_dict'], strict=False)

    # fastvit.fork_feat = True
    # fastvit.out_indices = [0, 2, 4, 6]

    img = torch.randn((1, 3, 224, 224))

    y = fastvit(img)
    print([i.shape for i in y])
    print(type(fastvit))


if __name__ == '__main__':
    main()
