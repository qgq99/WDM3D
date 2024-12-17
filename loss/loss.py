"""
-*- coding: utf-8 -*-
@Time    : 2024-11-21 17:01:17
@Author  : Qin Guoqing
@File    : loss.py
@Description : most of the code derived from WeakM3D.
"""


import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from utils.wdm3d_utils import create_module
from torch.nn import MSELoss, SmoothL1Loss
from utils.loss_tal_dual import ComputeLoss
from .loss_3d import calc_3d_loss, generate_data_for_loss
from .depth_loss import SiLogLoss
import pdb


G = globals()


class WDM3DLoss(nn.Module):

    def __init__(self, model=None, sample_roi_points=100, dim_prior=[[0.8, 1.8, 0.8], [0.6, 1.8, 1.8], [1.6, 1.8, 4.]], loss_weights=[1, 0.9, 0.9], inf_pixel_loss=0.001, *args, **config) -> None:
        """
        sample_roi_points: 每个实例采样多少个点云中的点
        dim_prior: 
        loss_weights: 依次为3dloss, depth loos, 2d bbox loss的权重
        inf_pixel_loss: 在计算depth loss时, 每个缺失深度信息的像素记多大的loss值
        """
        super().__init__()
        self.sample_roi_points = sample_roi_points
        self.dim_prior = dim_prior
        self.loss_weights = loss_weights
        self.inf_pixel_loss = inf_pixel_loss
        self.depth_loss = create_module(G, config, "depth_loss")
        self.bbox2d_loss = create_module(G, config, "bbox2d_loss", model=model.detector_2d)

        # self.siLogLoss = SiLogLoss()

    def forward(self, roi_points, bbox2d_pred, loss2d_feat, depth_pred, pred3d, bbox2d_gt, depth_gt, calibs):
        """
        TODO: depth_gt总包含值为inf的像素点, 表示该像素点确实深度, 需要特别处理
        """
        # pdb.set_trace()
        device = depth_pred.device
        batch_size = len(roi_points)

        obj_cnt_each_img = []
        """
        yolov9 loss中需求的label中image index实际为一个batch中的index
        """
        for i, v in enumerate(bbox2d_gt):
            # v[: 0] = i
            bbox2d_gt[i][:, 0] = i
            obj_cnt_each_img.append(v.shape[0])
        bbox2d_gt = torch.cat(bbox2d_gt)

        # pdb.set_trace()
        if isinstance(depth_gt, list):        
            depth_gt = torch.stack(depth_gt)

        
        depth_loss = self.depth_loss(depth_pred, depth_gt)
        bbox2d_loss = self.bbox2d_loss(loss2d_feat, bbox2d_gt)[0]


        loss_3d = torch.zeros(1, device=device)
        # pdb.set_trace()
        for i in range(batch_size):
            # """
            # 处理深度图中缺失深度信息的像素(值为inf)
            # 暂行方法：
            # 由于深度预测不会出现inf, 因此每个gt为inf的像素给定一个较小的损失值
            # TODO: 消融实验验证多种方法
            # """
            # # pdb.set_trace()
            # depth_inf_mask = depth_gt[i] == torch.inf
            # depth_gt[i][depth_inf_mask] = depth_pred[i][depth_inf_mask]
            # depth_loss = depth_loss + \
            #     self.depth_loss(depth_pred[i], depth_gt[i]) + \
            #     torch.sum(depth_inf_mask) * self.inf_pixel_loss

            # """
            # 处理预测到的目标数量和标签目标数量不同的情况
            # """
            mn_obj_cnt = min(len(bbox2d_pred[i]), obj_cnt_each_img[i])
            # bbox2d_loss = bbox2d_loss + self.bbox2d_loss(
            #     bbox2d_pred[i][:mn_obj_cnt, :4], bbox2d_gt[i][:mn_obj_cnt])

            # bbox2d_loss = bbox2d_loss

            # if len(bbox2d_pred[i]) != len(roi_points[i]):
            #     """
            #     len(roi_points[i])表示预测到的实例数， 若二者不相等, 说明2d框的预测值中有面积为0的框
            #     此时跳过该次计算并给定一个常数损失值
            #     """
            #     # pdb.set_trace()
            #     print("skip 3d loss")
            #     loss_3d = loss_3d + 0   # 应将0替换某一正值, 且配置为参数
            if mn_obj_cnt == 0:
                """
                gt无目标或检测结果无目标, 无从计算3d loss
                """
                print("gt无目标或检测结果无目标 0 3d loss")
                loss_3d = loss_3d + 0   # 应将0替换某一正值, 且配置为参数
            else:
                # pdb.set_trace()
                data = generate_data_for_loss(
                    roi_points[i], bbox2d_pred[i][:mn_obj_cnt], sample_roi_points=self.sample_roi_points, dim_prior=self.dim_prior)
                # pdb.set_trace()
                loss_3d = loss_3d + calc_3d_loss(
                    pred_3D=(pred3d[0][i][:mn_obj_cnt], pred3d[1][i][:mn_obj_cnt], pred3d[2][i][:mn_obj_cnt]),
                    batch_RoI_points=data["batch_RoI_points"],
                    bbox2d=bbox2d_pred[i][:mn_obj_cnt, :4],
                    batch_lidar_y_center=data["batch_lidar_y_center"],
                    batch_lidar_density=data["batch_lidar_density"],
                    batch_lidar_orient=data["batch_lidar_orient"],
                    batch_dim=data["batch_dim"].to(device),
                    P2=torch.tensor(calibs[i].P, device=device)
                )

        
        # depth loss偶尔出现极大值, 3e21等, 为避免极大值导致无法看出其他正常值的趋势, 暂屏蔽该情况
        # TODO: 考察depth loss出现极大值的原因
        # depth_loss if depth_loss < 1e15 else (depth_loss * 1e-15)
        
    
        total_loss = loss_3d * self.loss_weights[0] + depth_loss * self.loss_weights[1] + bbox2d_loss * self.loss_weights[2]
        # pdb.set_trace()
        return total_loss, loss_3d, depth_loss, bbox2d_loss
