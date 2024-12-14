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
from matplotlib import pyplot as plt


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


def plot_loss_curve(loss_datas, titles=[], xlabels=[], ylabels=[], output_dir="", show=False, save=True):
    assert len(loss_datas) == len(titles) == len(xlabels) == len(ylabels)

    plot_cnt = len(loss_datas)
    row = 2
    col = (plot_cnt + 1) // row

    fig, axs = plt.subplots(row, col)

    for i in range(plot_cnt):
        axs[i // col, i %
            col].plot(list(range(len(loss_datas[i]))), loss_datas[i])
        axs[i // col, i % col].set_title(titles[i])
        axs[i // col, i % col].set_xlabel(xlabels[i])
        axs[i // col, i % col].set_ylabel(ylabels[i])

    # 隐藏最后一个子图（空白占位，避免最后一个图显示异常）
    if row * col > plot_cnt:
        axs[row-1, col - 1].axis('off')
    plt.tight_layout()

    if save:
        plt.savefig(output_dir / "loss_curve.png")
    if show:
        plt.show()


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
    # pdb.set_trace()

    dataset = create_module(G, config, "dataset")
    dataloader = create_dataloader(dataset=dataset, batch_size=batch_size)

    batch_cnt = len(dataloader)  # 共有多少个batch
    # pdb.set_trace()
    model = WDM3D(config["model"]).to(device)

    loss_preocessor = create_module(G, config, "loss", model=model)

    # pdb.set_trace()
    optimizer = create_optimizer(model, config["optimizer"])

    """
    依次保存每个epoch的总loss, 平均总loss, 平均3dloss, 平均bbox2dloss, 平均depth loss
    """
    sum_epoch_losses, avg_epoch_losses, avg_epoch_3dlosses, avg_bbox2d_losses, avg_depth_losses = [], [], [], [], []

    model.train()
    with Timer("the hole training process", printer=logger.success):
        for epoch_idx in range(epoch):
            with Timer(f"epoch {epoch_idx+1}", printer=logger.success):
                cur_epoch_total_loss, cur_epoch_3dloss, cur_epoch_bbox2d_loss, cur_epoch_depth_loss = 0, 0, 0, 0
                for batch_idx, (img, targets, original_idx) in enumerate(dataloader):
                    # pdb.set_trace()
                    with Timer(f"batch [{batch_idx}]", work=False):
                        optimizer.zero_grad()
                        img = img.to(device)
                        targets = [t.to(device) for t in targets]
                        # pdb.set_trace()
                        bbox_2d, loss2d_feat, depth_pred, pseudo_LiDAR_points, pred = model(
                            img, targets)
                        # pdb.set_trace()
                        total_loss, loss_3d, depth_loss, bbox2d_loss = loss_preocessor(
                            roi_points=pseudo_LiDAR_points,
                            bbox2d_pred=bbox_2d,
                            loss2d_feat=loss2d_feat,
                            depth_pred=depth_pred,
                            pred3d=pred[1],
                            bbox2d_gt=[t.get_field("bbox2d_gt")
                                       for t in targets],
                            depth_gt=[t.get_field("depth_map")
                                      for t in targets],
                            calibs=[t.get_field("calib") for t in targets]
                        )
                        logger.info(
                            f"Epoch [{epoch_idx+1}/{epoch}], batch [{batch_idx+1}/{batch_cnt}], batch loss: [{total_loss.item()}], 3d loss: [{loss_3d.item()}], depth_loss: [{depth_loss.item()}], bbox2d_loss: [{bbox2d_loss.item()}], image index: {original_idx}")

                        cur_epoch_total_loss += total_loss.item()
                        cur_epoch_3dloss += loss_3d.item()
                        cur_epoch_bbox2d_loss += bbox2d_loss.item()
                        cur_epoch_depth_loss += depth_loss.item()

                        total_loss.backward()
                        optimizer.step()
                    # break

                sum_epoch_losses.append(cur_epoch_total_loss)
                avg_epoch_losses.append(cur_epoch_total_loss / batch_cnt)
                avg_epoch_3dlosses.append(cur_epoch_3dloss / batch_cnt)
                avg_bbox2d_losses.append(cur_epoch_bbox2d_loss / batch_cnt)
                avg_depth_losses.append(cur_epoch_depth_loss / batch_cnt)

    plot_loss_curve([sum_epoch_losses, avg_epoch_losses, avg_epoch_3dlosses, avg_bbox2d_losses, avg_depth_losses], titles=[
                    "sum_epoch_losses", "avg_epoch_losses", "avg_epoch_3dlosses", "avg_bbox2d_losses", "avg_depth_losses"],
                    xlabels=["epoch"] * 5, ylabels=["loss"] * 5, output_dir=output_dir)

    model_save_path = output_dir / "model_sd.pth"
    torch.save(model.state_dict(), model_save_path)
    logger.success(
        f"The final model state dict has been save to {model_save_path}")


if __name__ == '__main__':
    main(get_arg_parser().parse_args())
