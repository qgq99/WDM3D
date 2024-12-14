"""
-*- coding: utf-8 -*-
@Time    : 2024-11-09 20:46:07
@Author  : Qin Guoqing
@File    : temp.py
@Description : Description of this file
"""

from utils.wdm3d_utils import load_config, create_module, Timer
import torch
from model.model import WDM3D

G = globals()







def main():
    device = torch.device("cuda:0")
    cfg = load_config("/home/qinguoqing/project/WDM3D/config/exp/exp.yaml")
    cfg["model"]["ckpt"] = "/home/qinguoqing/project/WDM3D/output/train/check_effect_DA2_SiLogLoss_loss_2024-12-12_15_26_25/model_sd.pth"
    model = WDM3D(cfg["model"]).to(device)



if __name__ == '__main__':
    main()
