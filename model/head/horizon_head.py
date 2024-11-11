"""
-*- coding: utf-8 -*-
@Time    : 2024-11-11 14:17:01
@Author  : Qin Guoqing
@File    : horizon_head.py.py
@Description : horizon_head, code derived from MonoCD
"""

from torch import nn
from torch.nn import functional as F

class HorizonHead(nn.Module):
    
    def __init__(self, in_channel, mid_channel) -> None:
        """
        in_channel:
        mid_channel: 两个卷积层, mid_channel表示第一个卷积层的输出通道数
        """
        super().__init__()

        self.conv1 = nn.Conv2d(in_channel, mid_channel, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(mid_channel)
        self.conv2 = nn.Conv2d(mid_channel, 1, kernel_size=1, bias=False)


    
    def forward(self, x):
        return self.conv2(F.relu(self.norm(self.conv1(x))))

