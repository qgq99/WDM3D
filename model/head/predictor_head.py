"""
-*- coding: utf-8 -*-
@Time    : 2024-11-10 18:58:16
@Author  : Qin Guoqing
@File    : predictor_head.py.py
@Description : Description of this file
"""

import torch
from torch import nn
from torchvision.ops import roi_align



class WDM3DPredictorHead(nn.Module):
    """
    A predictor head to predict 3D bbox, derived from WeakM3D.
    """

    def __init__(self, channels=256) -> None:
        super().__init__()
        self.channels = channels

        self.location_xy = nn.Sequential(
            nn.Linear(self.channels * 7 * 7, 256),
            nn.ReLU(True),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, 2),
        )
        self.location_z = nn.Sequential(
            nn.Linear(self.channels * 7 * 7, 256),
            nn.ReLU(True),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, 1),
        )
        self.orientation_conf = nn.Sequential(
            nn.Linear(self.channels * 7 * 7, 256),
            nn.ReLU(True),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, 2),
        )

    
    def forward(self, feature: torch.Tensor, bbox: torch.Tensor):

        bbox = [item[:, :4] for item in bbox]

        if type(bbox) == list:
            f = roi_align(
                feature, [i/16 for i in bbox], (7, 7))
        else:
            f = roi_align(feature, [bbox/16], (7, 7))

        f = f.view(-1, self.channels * 7 * 7)

        location_xy = self.location_xy(f)
        location_xy = location_xy.view(-1, 2)

        location_z = self.location_z(f)
        orientation_conf = self.orientation_conf(f)

        return location_xy, location_z, orientation_conf