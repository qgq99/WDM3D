"""
-*- coding: utf-8 -*-
@Time    : 2024-11-27 14:38:27
@Author  : Qin Guoqing
@File    : train.py
@Description : Description of this file
"""

import torch
from model.model import WDM3D
from utils.wdm3d_utils import load_config, create_module, create_dataloader, Timer, create_optimizer, get_current_time, dump_config
from dataset.kitti.kitti import KITTIDataset
from loss import WDM3DLoss
from torchvision.transforms import Compose, ToTensor
# import cv2
import pdb
import numpy as np
import argparse
from loguru import logger
import time
import os
from pathlib import Path


G = globals()

trnasform = Compose([
    ToTensor()
])


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", default="", help="experiment name")
    parser.add_argument("--desc", default="",
                        help="description of this experiment")
    parser.add_argument(
        "--config", default="/home/qinguoqing/project/WDM3D/config/exp/exp.yaml")
    parser.add_argument("--epoch", default=10, type=int)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--CUDA_VISIBLE_DEVICES", default="0")
    parser.add_argument("--device", default="cuda:0")

    parser.add_argument("--output_dir", default="")

    return parser


def main(args):
    batch_size = args.batch_size
    device = torch.device(args.device)
    config = load_config(args.config)
    epoch = args.epoch
    output_dir = Path(args.output_dir)

    exp_meta_data = {
        "title": args.title,
        "desc": args.desc,
        "time": get_current_time(),
        "batch_size": batch_size,
        "epoch": epoch,
        "CUDA_VISIBLE_DEVICES": args.CUDA_VISIBLE_DEVICES,
        "device": args.device,
    }

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    """
    因为希望实验元数据显示在所保存的配置文件的最上方, 
    若先合并配置对象再dump, 会导致元数据显示在最下方, 所以分两次dump
    """
    dump_config(exp_meta_data, output_dir / "config.yaml")
    dump_config(config, output_dir / "config.yaml", is_append=True)

    logger.info(
        f"Config of this experiment has been saved to {output_dir / 'config.yaml'}.")
    pdb.set_trace()
    loss_preocessor = create_module(G, config, "loss")

    dataset = create_module(G, config, "dataset")
    dataloader = create_dataloader(dataset=dataset, batch_size=batch_size)

    model = WDM3D(config["model"]).to(device)
    optimizer = create_optimizer(model, config["optimizer"])

    for epoch_idx in range(epoch):
        for img, targets, original_idx in dataloader:
            img = img.to(device)
            targets = [t.to(device) for t in targets]
            # pdb.set_trace()
            with Timer("forward", work=False):
                bbox_2d, depth_pred, pseudo_LiDAR_points, pred = model(
                    img, targets)

            with Timer("loss process", work=False):
                total_loss, loss_3d, depth_loss, bbox2d_loss = loss_preocessor(
                    roi_points=pseudo_LiDAR_points,
                    bbox2d_pred=bbox_2d,
                    depth_pred=depth_pred,
                    pred3d=pred[1],
                    bbox2d_gt=[t.get_field("2d_bboxes") for t in targets],
                    depth_gt=[t.get_field("depth_map") for t in targets],
                    calibs=[t.get_field("calib") for t in targets]
                )
            logger.info(
                f"batch loss: {total_loss.item()}, 3d loss: {loss_3d}, depth_loss: {depth_loss}, bbox2d_loss: {bbox2d_loss}")

            with Timer("backward", work=False):
                total_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            # break
        break


if __name__ == '__main__':
    main(get_arg_parser().parse_args())
