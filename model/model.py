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
import pdb


class WDM3D(nn.Module):

  def __init__(self, config=None) -> None:
    super().__init__()
    G = globals()

    if isinstance(config, str):
      with open(config) as f:
        self.cfg = yaml.safe_load(f)["model"]
    else:
      self.cfg = config
    self.backbone = G[self.cfg["backbone"]["module"]](**self.cfg["backbone"]["params"])

    self.detector_2d = ""

