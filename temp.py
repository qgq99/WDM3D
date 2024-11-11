"""
-*- coding: utf-8 -*-
@Time    : 2024-11-09 20:46:07
@Author  : Qin Guoqing
@File    : temp.py.py
@Description : Description of this file
"""

import torch
from model.model import WDM3D
import pdb

def main():
    batch_size = 8
    h, w = 384, 1280

    # h = h + 14 - (h % 14)
    # w = w + 14 - (w % 14)

    device = torch.device("cuda:1")
    model = WDM3D(
        "/home/qinguoqing/project/WDM3D/config/WDM3D.yaml").to(device)

    images = torch.randn((batch_size, 3, h, w)).to(device)
    pe_img_comput = torch.randn((batch_size, h, w)).to(device)
    res = model(images, targets=pe_img_comput)
    print(res[0])


if __name__ == '__main__':
    main()
