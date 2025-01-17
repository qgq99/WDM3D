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
import open3d as o3d
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
    # points = points.reshape((-1, 3))
    # points = points.T

    points = points.reshape((3, -1))
    points = points.T

    cloud = calib.project_image_to_velo(points)
    valid = (cloud[:, 0] >= 0) & (cloud[:, 2] < max_high)
    return cloud[valid]


def depth_to_point_cloud(depth_image, intrinsics):
    """
    将深度图转换为点云
    :param depth_image: 深度图 (2D NumPy 数组)
    :param intrinsics: 相机内参 (Camera Intrinsics)
    :return: 点云 (open3d.geometry.PointCloud)
    """
    height, width = depth_image.shape
    # pdb.set_trace()
    # 获取相机内参
    fx, fy, cx, cy = intrinsics['fx'], intrinsics['fy'], intrinsics['cx'], intrinsics['cy']

    # 生成像素坐标网格
    x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))

    # 根据深度图计算对应的3D坐标
    depth_image = depth_image.numpy()
    # z_coords = depth_image / 1000.0  # 假设深度单位是毫米，转换为米
    z_coords = depth_image 
    x_coords = (x_coords - cx) * z_coords / fx
    y_coords = (y_coords - cy) * z_coords / fy

    y_coords *= -1

    # 将 3D 坐标转化为点 (x, y, z)
    # points = np.stack((x_coords, y_coords, z_coords), axis=-1)
    points = np.stack((y_coords, x_coords, z_coords), axis=-1)

    # 扁平化点云
    points = points.reshape((-1, 3))

    # 创建 Open3D 点云对象
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)

    return np.asarray(point_cloud.points)




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


def generate_pseudo_point_cloud_with_open3d(depth, calib:Calibration, bboxes):
    """
    将bbox范围内的像素点选出来, 并用选出来的像素点计算伪点云, 使用open3d作为计算工具
    """
    intrinsics = {
        'fx': calib.f_u,  # x轴焦距
        'fy': calib.f_v,  # y轴焦距
        'cx': calib.c_u,  # 光心x坐标
        'cy': calib.c_v,  # 光心y坐标
    }
    single_img_pseudo_roi_point_cloud = []
    for [x1, y1, x2, y2] in bboxes:
        single_img_pseudo_roi_point_cloud.append(depth_to_point_cloud(depth[y1: y2, x1: x2], intrinsics))
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
            f"Successfully created WDM3D model, model parameter count: {calc_model_params_count(self):.2f}M")

    def forward(self, x: torch.Tensor, targets=None):
        if self.training:
            return self.forward_train(x, targets)
        return self.forward_test()

        

    def forward_train(self, x: torch.Tensor, targets):
        b, c, h, w = x.shape
        device = x.device
        features = self.backbone(x)

        detector_2d_output = self.detector_2d(x)
        bbox_2d = non_max_suppression(
            detector_2d_output[0][0].detach(), conf_thres=0.1, max_det=20)
        bbox_2d = [i.detach() for i in bbox_2d]     # predicted bbox do not need gradient



        # pdb.set_trace()

        depth_pred, depth_feat = self.depther(features, h, w)

        depth_pred_detach = depth_pred.detach()

        pes =  [calc_pe(h, w, t.get_field("calib")) for t in targets]
        slope_maps = torch.stack([generate_slope_map(p, d.detach().cpu()) for p, d in zip(pes, depth_pred_detach)]).to(device)

        neck_output_feats, y, pe_mask, pe_slope_k_ori = self.neck(
            features, h, w, slope_maps)



        # pdb.set_trace()
        # bbox_2d = [torch.stack([random_bbox2d(device=device) for _ in range(6)]) for __ in range(b)]


        # pseudo_LiDAR_points = self.calc_pseudo_LiDAR_point(
        #     depth_pred, [t.get_field("calib") for t in targets])
        # pdb.set_trace()

        pseudo_LiDAR_points = self.calc_selected_pseudo_LiDAR_point(
            depth_pred_detach, bbox_2d, [t.get_field("calib") for t in targets], img_size=(h, w))

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

        pes =  [calc_pe(h, w, calib)]
        slope_maps = torch.stack([generate_slope_map(p, d.detach().cpu()) for p, d in zip(pes, depth_pred)]).to(device)

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
        return pred[1], bbox_2d

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



class WDM3DDepthOff(nn.Module):

    """
    A WDM3D model without depther inside it, but receive depth from outside, namely "depth off-line".
    """

    def __init__(self, config=None):
        super().__init__()
        self.backbone: nn.Module
        self.neck: nn.Module
        self.neck_fusion: nn.Module
        # self.depther: nn.Module
        # self.detector_2d: nn.Module
        self.head: nn.Module

        if isinstance(config, str):
            with open(config) as f:
                self.cfg = yaml.safe_load(f)["model"]
        else:
            self.cfg = config

        for prop in ["backbone", "neck", "neck_fusion", "head"]:
            setattr(self, prop, create_module(G, self.cfg, prop))

        # if type(self.detector_2d) == DetectionModel:
        #     # **if use yolov9 as the 2d detector, attach hyperparameters to it for its loss computation**
        #     self.detector_2d.hyp = load_config(
        #         "/home/qinguoqing/project/WDM3D/config/yolo/hyp.scratch-high.yaml", sub_cfg_keys=[])

        # ================================load pretrained weight======================================================
        # pdb.set_trace()
        if "ckpt" in config and os.path.exists(config["ckpt"]):
            self.load_state_dict(torch.load(config["ckpt"], weights_only=True))
            logger.success(f"Successfully loaded ckeckpoint: {config['ckpt']}")

        # if "detector_2d_ckpt" in config and os.path.exists(config["detector_2d_ckpt"]):
        #     self.detector_2d.load_state_dict(torch.load(
        #         config["detector_2d_ckpt"], weights_only=True))
        #     # self.detector_2d.eval()
        #     logger.success(
        #         f"Successfully loaded detector_2d_ckpt: {config['detector_2d_ckpt']}")

        if type(self.backbone) == FastViT and "backbone_ckpt" in config and os.path.exists(config["backbone_ckpt"]):
            self.backbone.load_state_dict(torch.load(
                config["backbone_ckpt"], weights_only=True), strict=False)
            logger.success(
                f"Successfully loaded backbone_ckpt: {config['backbone_ckpt']}")

        logger.success(
            f"Successfully created WDM3D model, model parameter count: {calc_model_params_count(self):.2f}M")
    
    def forward(self, x, depths, bbox_2d, calibs):
        b, c, h, w = x.shape
        device = x.device
        features = self.backbone(x)
        # pdb.set_trace()
        # pdb.set_trace()
        # detector_2d_output = self.detector_2d(x)
        # # pdb.set_trace()
        # bbox_2d = non_max_suppression(
        #     detector_2d_output[0][0].detach(), conf_thres=0.1, max_det=10)
        # pdb.set_trace()
        # bbox_2d = [i.detach() for i in bbox_2d]     # predicted bbox do not need gradient
        # pdb.set_trace()
        # bbox_2d = [i[i[:, -1] == 2] for i in bbox_2d]   # filter out Car objs
        
        # for i in range(len(bbox_2d)):
        #     bbox_2d[i][:, -1] = 0



        # depth_pred, depth_feat = self.depther(features, h, w)
        # pdb.set_trace()
        # pes =  [calc_pe(h, w, calib) for calib in calibs]
        # slope_maps = torch.stack([generate_slope_map(p, d.detach().cpu()) for p, d in zip(pes, depths)]).to(device)

        # # pdb.set_trace()
        # neck_output_feats, y, pe_mask, pe_slope_k_ori = self.neck(
        #     features, h, w, slope_maps)

        # pdb.set_trace()
        pseudo_LiDAR_points = self.calc_selected_pseudo_LiDAR_point_with_open3d(
            depths, bbox_2d, calibs, img_size=(h, w))

        # print(pseudo_LiDAR_points)

        # depth_aware_feats = self.neck_fusion(neck_output_feats)
        # pdb.set_trace()
        # pred = self.head(depth_aware_feats, bbox_2d)
        pred = self.head(features, bbox_2d)
        """
        detector_2d_output[1] is for yolov9 loss
        """
        return pred, pseudo_LiDAR_points

    
    def forward_test(self, x, depths, bbox_2d, calibs):
        b, c, h, w = x.shape
        device = x.device
        features = self.backbone(x)

        # pdb.set_trace()
        # detector_2d_output = self.detector_2d(x)
        # # pdb.set_trace()
        # bbox_2d = non_max_suppression(
        #     detector_2d_output[0][0].detach(), conf_thres=0.1, max_det=10)
        # bbox_2d = [i.detach() for i in bbox_2d]     # predicted bbox do not need gradient
        # pdb.set_trace()
        # bbox_2d = [i for i in bbox_2d if i[-1] == 2]

        # depth_pred, depth_feat = self.depther(features, h, w)
        # pdb.set_trace()
        # pes =  [calc_pe(h, w, calib) for calib in calibs]
        # slope_maps = torch.stack([generate_slope_map(p, d.detach().cpu()) for p, d in zip(pes, depths)]).to(device)

        # # pdb.set_trace()
        # neck_output_feats, y, pe_mask, pe_slope_k_ori = self.neck(
        #     features, h, w, slope_maps)

        # pdb.set_trace()
        # pseudo_LiDAR_points = self.calc_selected_pseudo_LiDAR_point_with_open3d(
        #     depths, bbox_2d, calibs, img_size=(h, w))

        # print(pseudo_LiDAR_points)

        depth_aware_feats = self.neck_fusion(features)

        pred = self.head(depth_aware_feats, bbox_2d)
        """
        detector_2d_output[1] is for yolov9 loss
        """
        return pred[1]
    
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
        tmp_depths = [d.detach().cpu() for d in depths]
        for d, calib, bbox in zip(tmp_depths, calibs, bboxes):
            # bbox = bbox.clone().detach().type(torch.int32).cpu()
            bbox = bbox.detach().type(torch.int32).cpu()
            # pdb.set_trace()
            pseudo_LiDAR_points.append(
                select_depth_and_project_to_points(d, calib, bbox[:, :4]))

        return pseudo_LiDAR_points
    

    def calc_selected_pseudo_LiDAR_point_with_open3d(self, depths: torch.Tensor, bboxes: list[np.ndarray], calibs: list[Calibration], img_size=(384, 1280)):
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
        tmp_depths = [d.detach().cpu() for d in depths]
        for d, calib, bbox in zip(tmp_depths, calibs, bboxes):
            # bbox = bbox.clone().detach().type(torch.int32).cpu()
            bbox = bbox.detach().type(torch.int32).cpu()
            # pdb.set_trace()
            pseudo_LiDAR_points.append(
                generate_pseudo_point_cloud_with_open3d(d, calib, bbox[:, :4]))

        return pseudo_LiDAR_points
