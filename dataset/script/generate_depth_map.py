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


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kitti_dir", default="/home/qinguoqing/dataset/kitti")

    return parser


def generate_depth_maps(split_dir: Path, filenames: list[str]):

    # 保存每个文件的原始深度的最大值最小值
    mx_mn_val_map = {}

    for name in tqdm(filenames, f"计算进度"):
        img = cv2.imread(split_dir / "image_2" / f"{name}.png")
        h, w, _ = img.shape
        calib = Calibration(split_dir / "calib" / f"{name}.txt")
        point_cloud = load_velo_scan(split_dir / "velodyne" / f"{name}.bin")
        coords, depths = calib.project_velo_to_image(point_cloud[:, :3])

        # 取出坐标没有超出图像坐标系范围的坐标
        valid_mask = (coords[:, 0] >= 0) & (
            coords[:, 0] < w-1) & (coords[:, 1] >= 0) & (coords[:, 1] < h-1)

        coords = np.round(coords[valid_mask]).astype(int)
        depths = depths[valid_mask]

        mnd, mxd = depths.min(), depths.max()
        mx_mn_val_map[name] = [mxd, mnd]
        depths = np.round((depths - mnd) / (mxd - mnd) * 255)
        depth_img = np.full((h, w), np.inf)
        depth_img[coords[:, 1], coords[:, 0]] = depths
        cv2.imwrite(str(split_dir / "depth" /
                    "depth_maps" / f"{name}.png"), depth_img)

    with open(str(split_dir / "depth" / f"max_min_val.txt"), "x") as f:
        for name in mx_mn_val_map:
            mx, mn = mx_mn_val_map[name]
            f.write(f"{name} {mx} {mn}\n")


def main(args):
    trainset_dir = Path(args.kitti_dir) / "train"
    testset_dir = Path(args.kitti_dir) / "test"

    # if not os.path.exists(trainset_dir / "depth"):
    #     os.mkdir(trainset_dir / "depth")
    # if not os.path.exists(testset_dir / "depth"):
    #     os.mkdir(testset_dir / "depth")

    if not os.path.exists(trainset_dir / "depth" / "depth_maps"):
        os.mkdir(trainset_dir / "depth" / "depth_maps")
    if not os.path.exists(testset_dir / "depth" / "depth_maps"):
        os.mkdir(testset_dir / "depth" / "depth_maps")

    trainset = [i.removesuffix(".png")
                for i in os.listdir(trainset_dir / "image_2")]
    testset = [i.removesuffix(".png")
               for i in os.listdir(testset_dir / "image_2")]

    # generate_depth_maps(trainset_dir, trainset)
    generate_depth_maps(testset_dir, testset)


if __name__ == '__main__':
    main(get_arg_parser().parse_args())
