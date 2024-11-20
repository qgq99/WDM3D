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
from utils.wdm3d_utils import create_module, calc_model_params_count
from utils.general import non_max_suppression
from dataset.kitti.kitti_utils import Calibration
from model.backbone import *
from model.detector_2d import *
from model.depther import *
from model.head import *
from model.neck import *
from model.layer import *
import pdb


G = globals()


def project_depth_to_points(calib, depth, max_high):
    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows))
    points = np.stack([c, r, depth])
    points = points.reshape((-1, 3))
    # points = points.T
    cloud = calib.project_image_to_velo(points)
    valid = (cloud[:, 0] >= 0) & (cloud[:, 2] < max_high)
    return cloud[valid]


class WDM3D(nn.Module):

    def __init__(self, config=None) -> None:
        super().__init__()

        self.backbone: nn.Module
        self.neck: nn.Module
        self.neck_fusion: nn.Module
        self.depther: nn.Module
        self.detector_2d: nn.Module
        self.head: nn.Module

        if isinstance(config, str):
            with open(config) as f:
                self.cfg = yaml.safe_load(f)["model"]
        else:
            self.cfg = config

        for prop in ["backbone", "neck", "neck_fusion", "depther", "detector_2d", "head"]:
            setattr(self, prop, create_module(G, self.cfg, prop))

        print(
            f"Successfully create WDM3D model, model parameter count: {calc_model_params_count(self):.2f}MB")

    def forward(self, x: torch.Tensor, targets=None):

        if self.training:
            return self.forward_train(x, targets)
        return self.forward_test(x)

    def forward_train(self, x: torch.Tensor, targets):
        b, c, h, w = x.shape
        features = self.backbone(x)
        # pdb.set_trace()
        neck_output_feats, y, pe_mask, pe_slope_k_ori = self.neck(
            features, h, w, torch.stack([t.get_field("slope_map") for t in targets]))

        detector_2d_output = self.detector_2d(x)
        bbox_2d = non_max_suppression(detector_2d_output[0])

        depth_pred, depth_feat = self.depther(neck_output_feats, h, w)

        pseudo_LiDAR_points = self.calc_pseudo_LiDAR_point(
            depth_pred, [t.get_field("calib") for t in targets])

        """
        因为depth_feat与neck_output_feats[0]的尺寸相同, 先做初步融合
        TODO: 验证相加, 相乘或更多融合操作的效果
        """
        neck_output_feats[0] = neck_output_feats[0] + depth_feat
        pdb.set_trace()

        depth_aware_feats = self.neck_fusion(neck_output_feats)

        pred = self.head(depth_aware_feats, bbox_2d)
        return bbox_2d, depth_pred, pseudo_LiDAR_points, neck_output_feats, depth_aware_feats, pred

    def forward_test(self, x):
        pass

    def calc_pseudo_LiDAR_point(self, depths: torch.Tensor, calibs: list[Calibration]):
        pseudo_LiDAR_points = []
        tmp_depths = depths.clone().detach().cpu()
        for d, calib in zip(tmp_depths, calibs):
            # pdb.set_trace()
            pseudo_LiDAR_points.append(
                project_depth_to_points(calib, d, max_high=100))

        return pseudo_LiDAR_points
