"""
-*- coding: utf-8 -*-
@Time    : 2024-11-09 13:22:23
@Author  : Qin Guoqing
@File    : generate_depth_map.py.py
@Description : Description of this file
"""

import argparse
from dataset.kitti.kitti_utils import Calibration, load_velo_scan
from pathlib import Path
import os
import cv2
import numpy as np
from tqdm import tqdm
from scipy.interpolate import griddata
import pdb


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kitti_dir", default="/home/qinguoqing/dataset/kitti")
    parser.add_argument("--interpolate", default=True, type=bool)

    return parser


def generate_depth_maps(split_dir: Path, filenames: list[str], interpolate=False):
    """
    interpolate: 是否对计算得到的深度图应用插值以获取更丰富的深度信息, 但插值可能导致深度信息失真
    """

    # 保存每个文件的原始深度的最大值最小值
    mx_mn_val_map = {}

    for name in tqdm(filenames, f"计算进度"):
        img = cv2.imread(split_dir / "image_2" / f"{name}.png")
        h, w, _ = img.shape
        calib = Calibration(split_dir / "calib" / f"{name}.txt")
        point_cloud = load_velo_scan(split_dir / "velodyne" / f"{name}.bin")
        coords, depths = calib.project_velo_to_image(point_cloud[:, :3])

        # 取出坐标没有超出图像坐标系范围且深度值为非负数(表示在相机前面)的坐标
        valid_mask = (coords[:, 0] >= 0) & (
            coords[:, 0] < w-1) & (coords[:, 1] >= 0) & (coords[:, 1] < h-1) & (depths >= 0)

        coords = np.round(coords[valid_mask]).astype(int)
        depths = depths[valid_mask]

        mnd, mxd = depths.min(), depths.max()
        mx_mn_val_map[name] = [mxd, mnd]
        """
        将深度值归一化到[1-255], 同时记录最大值和最小值
        若像素值为0, 表示此处缺失深度值
        待读取时, 可将像素值还原为深度值
        """
        depths = np.round((depths - mnd) / (mxd - mnd) * 254 + 1)

        depth_img = np.full((h, w), 0)

        if interpolate:
            grid_X, grid_y = np.meshgrid(np.arange(w), np.arange(h))
            depth_img_interp = griddata(
                coords, depths, (grid_X, grid_y), method="linear", fill_value=np.nan)
            depth_img = np.where(np.isnan(depth_img_interp),
                                 depth_img, depth_img_interp)
        else:
            depth_img[coords[:, 1], coords[:, 0]] = depths

        save_path = str(split_dir / "depth" /
                        f"depth_maps{'_interp' if interpolate else ''}" / f"{name}.png")
        cv2.imwrite(save_path, depth_img)

    with open(str(split_dir / "depth" / f"max_min_val.txt"), "w") as f:
        for name in mx_mn_val_map:
            mx, mn = mx_mn_val_map[name]
            f.write(f"{name} {mx} {mn}\n")


def main(args):
    trainset_dir = Path(args.kitti_dir) / "train"
    testset_dir = Path(args.kitti_dir) / "test"

    if not os.path.exists(trainset_dir / "depth" / "depth_maps"):
        os.mkdir(trainset_dir / "depth" / "depth_maps")
    if not os.path.exists(testset_dir / "depth" / "depth_maps"):
        os.mkdir(testset_dir / "depth" / "depth_maps")

    if args.interpolate:
        if not os.path.exists(trainset_dir / "depth" / "depth_maps_interp"):
            os.mkdir(trainset_dir / "depth" / "depth_maps_interp")
        if not os.path.exists(testset_dir / "depth" / "depth_maps_interp"):
            os.mkdir(testset_dir / "depth" / "depth_maps_interp")

    trainset = [i.removesuffix(".png")
                for i in os.listdir(trainset_dir / "image_2")]
    testset = [i.removesuffix(".png")
               for i in os.listdir(testset_dir / "image_2")]

    generate_depth_maps(trainset_dir, trainset, args.interpolate)
    generate_depth_maps(testset_dir, testset, args.interpolate)


if __name__ == '__main__':
    main(get_arg_parser().parse_args())
