"""
-*- coding: utf-8 -*-
@Time    : 2024-11-27 14:38:27
@Author  : Qin Guoqing
@File    : train.py
@Description : Description of this file
"""

import torch
from model.model import WDM3D
from utils.wdm3d_utils import load_config, create_module, create_dataloader, calc_model_params_count
from dataset.kitti.kitti import KITTIDataset
from loss import WDM3DLoss
from torchvision.transforms import Compose, ToTensor
# import cv2
import pdb
import numpy as np
import argparse


G = globals()

trnasform = Compose([
    ToTensor()
])



def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="/home/qinguoqing/project/WDM3D/config/exp/exp.yaml")
    parser.add_argument("--epoch", default=10, type=int)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--device", default="cuda:0")

    return parser



def main(args):
    batch_size = args.batch_size
    device = torch.device(args.device)
    config = load_config(args.config)
    epoch = args.epoch

    # pdb.set_trace()

    loss_preocessor = create_module(G, config, "loss")

    dataset = create_module(G, config, "dataset")
    dataloader = create_dataloader(dataset=dataset, batch_size=batch_size)

    model = WDM3D(config["model"]).to(device)

    for epoch_idx in range(epoch):
        for img, targets, original_idx in dataloader:
            img = img.to(device)
            targets = [t.to(device) for t in targets]
            pdb.set_trace()

            bbox_2d, depth_pred, pseudo_LiDAR_points, pred = model(img, targets)

            batch_loss = loss_preocessor(
                roi_points=pseudo_LiDAR_points,
                bbox2d_pred= bbox_2d,
                depth_pred=depth_pred,
                pred3d=pred[1],
                bbox2d_gt=[t.get_field("2d_bboxes") for t in targets],
                depth_gt=[t.get_field("depth_map") for t in targets],
                calibs=[t.get_field("calib") for t in targets]
            )

            print()
            break
        break


if __name__ == '__main__':
    main(get_arg_parser().parse_args())
