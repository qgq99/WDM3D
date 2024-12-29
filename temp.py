"""
-*- coding: utf-8 -*-
@Time    : 2024-11-09 20:46:07
@Author  : Qin Guoqing
@File    : temp.py
@Description : Description of this file
"""

from utils.wdm3d_utils import load_config, create_module, Timer
import torch
from model.backbone.vit.dinov2 import DINOv2
import pdb
G = globals()





def main():
    device = torch.device("cuda:1")
    # cfg = load_config("/home/qinguoqing/project/WDM3D/config/data/data.yaml", sub_cfg_keys=[])
    vits = DINOv2("vits").to(device)
    img = torch.randn((4, 3, 378, 1274), device=device)
    for _ in range(1000):
        feats = vits.get_intermediate_layers(img, [2, 5, 8, 11], return_class_token=True, reshape=True)
    print(feats)

if __name__ == '__main__':
    main()
