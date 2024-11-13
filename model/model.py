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
from utils.wdm3d_utils import create_module
from model.backbone import *
from model.detector_2d import *
from model.depther import *
from model.head import *
from model.neck import *
import pdb


G = globals()


class WDM3D(nn.Module):

    def __init__(self, config=None) -> None:
        super().__init__()

        self.backbone: nn.Module
        self.neck: nn.Module
        self.depther: nn.Module
        self.detector_2d: nn.Module
        self.head: nn.Module

        if isinstance(config, str):
            with open(config) as f:
                self.cfg = yaml.safe_load(f)["model"]
        else:
            self.cfg = config

        for prop in ["backbone", "neck", "depther", "detector_2d", "head"]:
            setattr(self, prop, create_module(G, self.cfg, prop))

    def forward(self, x: torch.Tensor, targets=None):
        if self.training:
            return self.forward_train(x, targets)
        return self.forward_test(x)

    def forward_train(self, x: torch.Tensor, targets):
        b, c, h, w = x.shape
        features = self.backbone(x)

        pred = self.depther(features)

        return pred

    def forward_test(self, x):
        pass
