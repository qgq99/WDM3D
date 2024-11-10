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
from model.depther import *
from model.head import *
from model.neck import *
import pdb


G = globals()

class WDM3D(nn.Module):

    

    def __init__(self, config=None) -> None:
        super().__init__()

        self.backbone: any
        self.neck: any
        self.depther: any
        self.detector_2d: any
        self.head: any

        if isinstance(config, str):
            with open(config) as f:
                self.cfg = yaml.safe_load(f)["model"]
        else:
            self.cfg = config

        modules_names = ["backbone", "neck", "depther", "detector_2d", "head"]
        for prop in modules_names:
            setattr(self, prop, G[self.cfg[prop]["module"]](
                **self.cfg[prop]["params"]))

