"""
-*- coding: utf-8 -*-
@Time    : 2024-11-09 20:38:33
@Author  : Qin Guoqing
@File    : model.py
@Description : Description of this file
"""

import yaml
import torch
from torch import nn
from utils.wdm3d_utils import create_module, calc_model_params_count, random_bbox2d, load_config
from utils.general import non_max_suppression
from dataset.kitti.kitti_utils import Calibration
from model.backbone import *
from model.detector_2d import *
from model.depther import *
from model.head import *
from model.neck import *
from model.layer import *
from loguru import logger
import pdb


G = globals()

def find_k(depth_map, pe):
    """
    copy-paste from code of GEDepth
    """
    a = 1.65 / pe
    b = 1.65 / depth_map
    k = b + a
    return k

def calc_pe(h, w, calib: Calibration):
    """
    calculate the pre-computed ground embeding, the following calculation process is acoording to paper "GEDepth"
    h: height of the current image
    w: width of the current image
    calib: calibration of the current image
    """
    u, v = np.meshgrid(range(w), range(h), indexing="xy")

    P2, R0_rect, Tr_velo_to_cam = calib.P, calib.R0, calib.V2C

    K = P2[:, 0:3]
    R = R0_rect
    T = Tr_velo_to_cam[:, 3]
    A = np.linalg.inv(K @ R)
    B = np.linalg.inv(R) @ (- T)
    zc = (1.65 - B[1]) / (A[1, 0] * u + A[1, 1] * v + A[1, 2])

    return zc

def generate_slope_map(pe, depth_map):
    """
    !!!
    the code is from GEDepth, its meaning is not so clear,
    guessing the k(ie. k-img in the original code) refer to slope map mentioned in its paper.
    !!!
    """
    valid_mask = depth_map == 0
    k = find_k(depth_map, pe)
    k = np.around(np.rad2deg(np.arctan(k)))
    k[k > 5] = 5
    k[k < -5] = -5
    k[valid_mask] = 255

    return k









def project_depth_to_points(calib, depth, max_high):
    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows))
    points = np.stack([c, r, depth])
    points = points.reshape((-1, 3))
    # points = points.T
    cloud = calib.project_image_to_velo(points)
    valid = (cloud[:, 0] >= 0) & (cloud[:, 2] < max_high)
    return cloud[valid]


def select_depth_and_project_to_points(depth, calib, bboxes):
    """
    将bbox范围内的像素点选出来, 并用选出来的像素点计算为点云, 核心逻辑同project_depth_to_points
    """
    # single_img_pseudo_point_cloud = np.zeros((0, 3))
    single_img_pseudo_roi_point_cloud = []
    for [x1, y1, x2, y2] in bboxes:
        c, r = np.meshgrid(np.arange(x1, x2), np.arange(y1, y2))
        # pdb.set_trace()
        points = np.stack([c, r, depth[y1: y2, x1: x2]]).reshape((-1, 3))
        # pdb.set_trace()
        if 0 not in points.shape:
            cloud = calib.project_image_to_velo(points)
            # pdb.set_trace()
            single_img_pseudo_roi_point_cloud.append(cloud)
    # return single_img_pseudo_point_cloud
    return single_img_pseudo_roi_point_cloud


def clamp_bboxes(bboxes: list[torch.Tensor], h: int, w: int):
    """
    将bbox坐标值clamp到图像范围内
    bboxes: 
    """
    clamped_bboxes = []

    for bbox in bboxes:
        # 使用torch.clamp限制边界框坐标
        clamped_bbox = bbox.clone()  # 创建一个副本以避免修改原始tensor
        clamped_bbox[:, 0] = torch.clamp(bbox[:, 0], min=0, max=w-1)  # x1
        clamped_bbox[:, 1] = torch.clamp(bbox[:, 1], min=0, max=h-1)  # y1
        clamped_bbox[:, 2] = torch.clamp(bbox[:, 2], min=0, max=w-1)  # x2
        clamped_bbox[:, 3] = torch.clamp(bbox[:, 3], min=0, max=h-1)  # y2

        clamped_bboxes.append(clamped_bbox)

    return clamped_bboxes


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

        if type(self.detector_2d) == DetectionModel:
            # **if use yolov9 as the 2d detector, attach hyperparameters to it for its loss computation**
            self.detector_2d.hyp = load_config(
                "/home/qinguoqing/project/WDM3D/config/yolo/hyp.scratch-high.yaml", sub_cfg_keys=[])

        # ================================load pretrained weight======================================================
        # pdb.set_trace()
        if "ckpt" in config and os.path.exists(config["ckpt"]):
            self.load_state_dict(torch.load(config["ckpt"], weights_only=True))
            logger.success(f"Successfully loaded ckeckpoint: {config['ckpt']}")

        if "detector_2d_ckpt" in config and os.path.exists(config["detector_2d_ckpt"]):
            self.detector_2d.load_state_dict(torch.load(
                config["detector_2d_ckpt"], weights_only=True))
            logger.success(
                f"Successfully loaded detector_2d_ckpt: {config['detector_2d_ckpt']}")

        if type(self.backbone) == FastViT and "backbone_ckpt" in config and os.path.exists(config["backbone_ckpt"]):
            self.backbone.load_state_dict(torch.load(
                config["backbone_ckpt"], weights_only=True), strict=False)
            logger.success(
                f"Successfully loaded backbone_ckpt: {config['backbone_ckpt']}")

        logger.success(
            f"Successfully created WDM3D model, model parameter count: {calc_model_params_count(self):.2f}MB")

    def forward(self, x: torch.Tensor, targets=None):
        if self.training:
            return self.forward_train(x, targets)
        return self.forward_test()

        

    def forward_train(self, x: torch.Tensor, targets):
        b, c, h, w = x.shape
        device = x.device
        features = self.backbone(x)
        # pdb.set_trace()
        neck_output_feats, y, pe_mask, pe_slope_k_ori = self.neck(
            features, h, w, torch.stack([t.get_field("slope_map") for t in targets]))

        # pdb.set_trace()
        detector_2d_output = self.detector_2d(x)
        # pdb.set_trace()
        bbox_2d = non_max_suppression(
            detector_2d_output[0][0].detach(), conf_thres=0.1, max_det=20)
        
        bbox_2d = [i.detach() for i in bbox_2d]     # predicted bbox do not need gradient

        # pdb.set_trace()
        # bbox_2d = [torch.stack([random_bbox2d(device=device) for _ in range(6)]) for __ in range(b)]

        depth_pred, depth_feat = self.depther(neck_output_feats, h, w)

        # pseudo_LiDAR_points = self.calc_pseudo_LiDAR_point(
        #     depth_pred, [t.get_field("calib") for t in targets])
        # pdb.set_trace()

        pseudo_LiDAR_points = self.calc_selected_pseudo_LiDAR_point(
            depth_pred, bbox_2d, [t.get_field("calib") for t in targets], img_size=(h, w))

        """
        因为depth_feat与neck_output_feats[0]的尺寸相同, 先做初步融合
        TODO: 验证相加, 相乘或更多融合操作的效果
        """
        neck_output_feats[0] = neck_output_feats[0] + depth_feat

        depth_aware_feats = self.neck_fusion(neck_output_feats)

        pred = self.head(depth_aware_feats, bbox_2d)
        """
        detector_2d_output[1] is for yolov9 loss
        """
        return bbox_2d, detector_2d_output[1], depth_pred, pseudo_LiDAR_points, pred

    def forward_test(self, x, calib: Calibration):
        b, c, h, w = x.shape
        device = x.device
        features = self.backbone(x)

        # pdb.set_trace()
        detector_2d_output = self.detector_2d(x)
        # pdb.set_trace()
        bbox_2d = non_max_suppression(
            detector_2d_output[0][0].detach(), conf_thres=0.1, max_det=20)
        bbox_2d = [i.detach() for i in bbox_2d]     # predicted bbox do not need gradient



        depth_pred, depth_feat = self.depther(features, h, w)

        pe = calc_pe(h, w, calib)

        slope_maps = torch.stack([generate_slope_map(pe, d) for d in depth_pred])

        # pdb.set_trace()
        neck_output_feats, y, pe_mask, pe_slope_k_ori = self.neck(
            features, h, w, slope_maps)



        # pdb.set_trace()
        # bbox_2d = [torch.stack([random_bbox2d(device=device) for _ in range(6)]) for __ in range(b)]



        # pseudo_LiDAR_points = self.calc_pseudo_LiDAR_point(
        #     depth_pred, [t.get_field("calib") for t in targets])
        # pdb.set_trace()

        # pseudo_LiDAR_points = self.calc_selected_pseudo_LiDAR_point(
        #     depth_pred, bbox_2d, [t.get_field("calib") for t in targets], img_size=(h, w))

        """
        因为depth_feat与neck_output_feats[0]的尺寸相同, 先做初步融合
        TODO: 验证相加, 相乘或更多融合操作的效果
        """
        neck_output_feats[0] = neck_output_feats[0] + depth_feat

        depth_aware_feats = self.neck_fusion(neck_output_feats)

        pred = self.head(depth_aware_feats, bbox_2d)
        """
        detector_2d_output[1] is for yolov9 loss
        """
        return pred

    def calc_selected_pseudo_LiDAR_point(self, depths: torch.Tensor, bboxes: list[np.ndarray], calibs: list[Calibration], img_size=(384, 1280)):
        """
        将bbox范围内的像素点选出来, 然后只用这些点计算伪点云
        depths: [bs, h, w]
        bboxes: [n, k], k >= 4, [x1, y1, x2, y2, ...]
        img_size: 图像尺寸, (h, w), 预测得到的bbox可能坐标超出图像范围, 需要clamp
        """
        # pdb.set_trace()

        bboxes = clamp_bboxes(bboxes, h=img_size[0], w=img_size[1])

        pseudo_LiDAR_points = []
        # tmp_depths = depths.clone().detach().cpu()
        tmp_depths = depths.detach().cpu()
        for d, calib, bbox in zip(tmp_depths, calibs, bboxes):
            # bbox = bbox.clone().detach().type(torch.int32).cpu()
            bbox = bbox.detach().type(torch.int32).cpu()
            # pdb.set_trace()
            pseudo_LiDAR_points.append(
                select_depth_and_project_to_points(d, calib, bbox[:, :4]))

        return pseudo_LiDAR_points

    def calc_pseudo_LiDAR_point(self, depths: torch.Tensor, calibs: list[Calibration]):
        """
        计算伪点云, 每个像素点都有一个深度, 因此每个像素对应一个点云中的点, shape[384, 1280]的深度图计算得到[491250, 3]的点云数据
        """
        pseudo_LiDAR_points = []
        # tmp_depths = depths.clone().detach().cpu()
        tmp_depths = depths.cpu()
        for d, calib in zip(tmp_depths, calibs):
            pdb.set_trace()
            pseudo_LiDAR_points.append(
                project_depth_to_points(calib, d, max_high=100))

        return pseudo_LiDAR_points
