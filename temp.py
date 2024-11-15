"""
-*- coding: utf-8 -*-
@Time    : 2024-11-09 20:46:07
@Author  : Qin Guoqing
@File    : temp.py.py
@Description : Description of this file
"""

# import torch
# from model.model import WDM3D
from utils.wdm3d_utils import load_config, create_module, create_dataloader
from dataset.kitti.kitti import KITTIDataset
import cv2
import pdb
import numpy as np

G = globals()


def main():
    # batch_size = 8
    # h, w = 384, 1280

    # # h = h + 14 - (h % 14)
    # # w = w + 14 - (w % 14)

    # device = torch.device("cuda:1")
    # model = WDM3D(
    #     "/home/qinguoqing/project/WDM3D/config/WDM3D.yaml").to(device)

    # images = torch.randn((batch_size, 3, h, w)).to(device)
    # pe_img_comput = torch.randn((batch_size, h, w)).to(device)
    # res = model(images, targets=pe_img_comput)

    # print(res[0])

    config = load_config()
    pdb.set_trace()
    dataset = create_module(G, config, "dataset")
    loader = create_dataloader(dataset, 4, False, 1)

    for batch in loader:
        print(batch["target"][0].get_field("depth_map").shape)
        break


if __name__ == '__main__':
    main()
