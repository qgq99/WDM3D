"""
-*- coding: utf-8 -*-
@Time    : 2024-11-09 20:46:07
@Author  : Qin Guoqing
@File    : temp.py
@Description : Description of this file
"""

# from utils.wdm3d_utils import load_config, create_module, Timer
import torch
from model.model import project_depth_to_points
from dataset.kitti.kitti_utils import Calibration
import pdb
import cv2
import numpy as np
# from tqdm import tqdm
G = globals()


# def calc_pseudo_LiDAR_point(depth: torch.Tensor, calib: list[Calibration]):
#     """
#     计算伪点云, 每个像素点都有一个深度, 因此每个像素对应一个点云中的点, shape[384, 1280]的深度图计算得到[491250, 3]的点云数据
#     """
#     pseudo_LiDAR_points = []
#     # tmp_depths = depths.clone().detach().cpu()
#     # tmp_depths = depth.cpu()
#     # for d, calib in zip(tmp_depths, calibs):
#     pdb.set_trace()
#     pseudo_LiDAR_points.append(
#         project_depth_to_points(calib, tmp_depths, max_high=100))

#     return pseudo_LiDAR_points


def main():
    # device = torch.device("cuda:1")
    # cfg = load_config("/home/qinguoqing/project/WDM3D/config/exp/exp_depth_off.yaml")
    depth_filepath = "/home/qinguoqing/dataset/kitti/train/depth/depth_maps_pred/000010.png"
    calib = Calibration(
        "/home/qinguoqing/dataset/kitti/train/calib/000010.txt")

    print(calib.f_u, calib.f_v, calib.c_u, calib.c_v)
    depth_map = cv2.imread(depth_filepath, -1)
    depth_map = np.where(depth_map > 0, ((depth_map - 1) / 254) *
                   (11.499794960021973 - 0) + 0, depth_map)  # 归一化的逆操作
    
    res = project_depth_to_points(calib, depth_map, 1)
    res = np.concatenate((res, np.zeros((res.shape[0], 1))), axis=1)
    res = res.astype(np.float32)
    

    np.save("000010_depth.npy", depth_map)
    with open('pseudo_lidar.bin', 'wb') as f:
        res.tofile(f)

    # print(res.shape)


if __name__ == '__main__':
    main()
