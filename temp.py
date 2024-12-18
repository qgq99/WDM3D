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
from model.detector_2d import *

G = globals()


def main():
    device = torch.device("cuda:0")
    # cfg = load_config("/home/qinguoqing/project/WDM3D/config/yolo/yolov9-s.yaml", sub_cfg_keys=[])

    yolo = DetectionModel(
        "/home/qinguoqing/project/WDM3D/config/yolo/yolov9-s.yaml")

    # # print(yolo)
    yolov9_sd = torch.load(
        "/home/qinguoqing/project/WDM3D/weight/model_sd.pth", weights_only=True)

    yolo.load_state_dict(yolov9_sd)
    # print(yolov9_sd.keys())
    print(yolo)


if __name__ == '__main__':
    main()
