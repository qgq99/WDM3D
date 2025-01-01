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
from torch.nn import functional as F
import pdb


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
        b, c, h, w = feature.shape
        bbox = [item[:, :4] for item in bbox]
    
        location_xy, location_z, orientation_conf = [], [], []
        for i in range(b):
            f = roi_align(feature[i].unsqueeze(0), [bbox[i] / 32], (7, 7))  # 除以的数值为feature的下采样倍率

            print(f"{torch.sum(f == 0) / torch.numel(f)}, {torch.sum(f == 0)}, {torch.numel(f)}")
            f = f.view(-1, self.channels * 7 * 7)
            location_xy.append(self.location_xy(f))
            location_z.append(self.location_z(f))
            print(location_z[-1])
            orientation_conf.append(self.orientation_conf(f))


        # pdb.set_trace()
        return location_xy, location_z, orientation_conf
