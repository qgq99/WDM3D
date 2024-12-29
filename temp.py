"""
-*- coding: utf-8 -*-
@Time    : 2024-11-09 20:46:07
@Author  : Qin Guoqing
@File    : temp.py
@Description : Description of this file
"""

# from utils.wdm3d_utils import load_config, create_module, Timer
import torch
# from model.backbone.vit.dinov2 import DINOv2
from model.depther.depth_anything_v2.dpt import DepthAnythingV2
import pdb
import cv2
from tqdm import tqdm
G = globals()





def main():
    device = torch.device("cuda:1")
    # cfg = load_config("/home/qinguoqing/project/WDM3D/config/data/data.yaml", sub_cfg_keys=[])
    model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}
    encoder = 'vits' # or 'vits', 'vitb', 'vitg'

    model = DepthAnythingV2(**model_configs[encoder]).to(device)

    model.load_state_dict(torch.load(f'/home/qinguoqing/project/WDM3D/weight/depth_anything_v2_vits.pth', map_location='cpu'))

    model.eval()

    raw_img = cv2.imread(f'/home/qinguoqing/dataset/kitti/train/image_2/000000.png')
    # raw_img = torch.tensor(raw_img, device=device)
    depth = model.infer_image(raw_img) # HxW raw depth map in numpy

    print(depth)


if __name__ == '__main__':
    main()
