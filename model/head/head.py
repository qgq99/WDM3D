"""
-*- coding: utf-8 -*-
@Time    : 2024-11-11 14:29:09
@Author  : Qin Guoqing
@File    : head.py.py
@Description : Description of this file
"""




from torch import nn
from utils.create_module import create_module
from .horizon_head import HorizonHead
from .predictor_head import WDM3DPredictorHead
import pdb

G = globals()

class WDM3DHead(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

        self.horizon_head = create_module(G, kwargs, "horizon_head")
        self.predictor_head = create_module(G, kwargs, "predictor_head")
    


    def forward(self, x):
        pass