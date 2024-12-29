"""
-*- coding: utf-8 -*-
@Time    : 2024-12-12 14:30:09
@Author  : Qin Guoqing
@File    : depth_loss.py
@Description : loss computation for depth, code derived from Depth-anything v2.
"""

import torch
import torch.nn as nn
import pdb

class SiLogLoss(nn.Module):
    def __init__(self, lambd=0.5, epsilon=1e-8):
        super().__init__()
        self.lambd = lambd
        self.epsilon = epsilon

    def forward(self, pred, target):
        # pdb.set_trace()
        # 屏蔽零值, 避免计算过程出现inf或nan
        pred = torch.clamp(pred, min=self.epsilon)
        target = torch.clamp(target, min=self.epsilon)

        # 计算预测值和目标值的对数差
        diff_log = torch.log1p(target) - torch.log1p(pred)
        
        # 计算损失
        loss = torch.sqrt(torch.pow(diff_log, 2).mean() - 
                          self.lambd * torch.pow(diff_log.mean(), 2)) * 100

        return loss