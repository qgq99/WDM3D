"""
-*- coding: utf-8 -*-
@Time    : 2024-11-09 20:38:33
@Author  : Qin Guoqing
@File    : model.py.py
@Description : Description of this file
"""

import yaml
import torch
from torch import nn
from model.backbone import *
from model.detector_2d import *
import pdb


G = globals()


class WDM3D(nn.Module):

    def __init__(self, config=None) -> None:
        super().__init__()

        if isinstance(config, str):
            with open(config) as f:
                self.cfg = yaml.safe_load(f)["model"]
        else:
            self.cfg = config
        self.backbone = G[self.cfg["backbone"]["module"]](
            **self.cfg["backbone"]["params"])

        # yolob9的parse_model会打印所创建的每一层的信息
        self.detector_2d = G[self.cfg["detecor_2d"]["module"]](
            **self.cfg["detecor_2d"]["params"])
