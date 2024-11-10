"""
-*- coding: utf-8 -*-
@Time    : 2024-11-09 20:46:07
@Author  : Qin Guoqing
@File    : temp.py.py
@Description : Description of this file
"""

import torch
from model.model import WDM3D


device = torch.device("cuda:0")


model = WDM3D("/home/qinguoqing/project/WDM3D/config/WDM3D.yaml").to(device)



print(model.neck)



