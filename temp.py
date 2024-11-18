"""
-*- coding: utf-8 -*-
@Time    : 2024-11-09 20:46:07
@Author  : Qin Guoqing
@File    : temp.py.py
@Description : Description of this file
"""

import torch
from model.model import WDM3D
from utils.wdm3d_utils import load_config, create_module, create_dataloader, calc_model_params_count
from dataset.kitti.kitti import KITTIDataset
from torchvision.transforms import Compose, ToTensor
# import cv2
import pdb
import numpy as np

G = globals()


trnasform = Compose([
    ToTensor()
])


def main():
    batch_size = 8
    h, w = 384, 1280
    device = torch.device("cuda:1")

    config = load_config("/home/qinguoqing/project/WDM3D/config/exp/exp.yaml")
    dataset = create_module(G, config, "dataset")

    dataloader = create_dataloader(dataset=dataset, batch_size=4)
    model = WDM3D(config["model"]).to(device)
    for img, targets, original_idx in dataloader:
        img = img.to(device)
        targets = [t.to(device) for t in targets]
        pdb.set_trace()

        res = model(img, targets)
        print(res)
        break

    # model = WDM3D(``
    #     "/home/qinguoqing/project/WDM3D/config/model/WDM3D.yaml").to(device)

    # images = torch.randn((batch_size, 3, h, w)).to(device)
    # pe_img_comput = torch.randn((batch_size, h, w)).to(device)
    # res = model(images, targets=pe_img_comput)
    # pdb.set_trace()
    # print(res)


if __name__ == '__main__':
    main()
