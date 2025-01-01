"""
-*- coding: utf-8 -*-
@Time    : 2024-11-09 20:46:07
@Author  : Qin Guoqing
@File    : temp.py
@Description : Description of this file
"""

# from utils.wdm3d_utils import load_config, create_module, Timer, get_current_time
import torch
# from model.model import project_depth_to_points
from model.detector_2d.yolov9.yolo import DetectionModel
# from dataset.kitti.kitti_utils import Calibration
import pdb
import cv2
from utils.general import non_max_suppression
# import numpy as np
# from tqdm import tqdm
G = globals()




def main():
    # device = torch.device("cuda:1")
    # cfg = load_config("/home/qinguoqing/project/WDM3D/config/exp/exp_depth_off.yaml")
    yolo = DetectionModel("/home/qinguoqing/project/WDM3D/config/yolo/yolov9-s.yaml")
    yolo.load_state_dict(torch.load("/home/qinguoqing/project/WDM3D/weight/yolov9-s-sd.pth", weights_only=True))

    img = cv2.imread("/home/qinguoqing/dataset/kitti/train/image_2/000003.png")

    img = cv2.resize(img, (1280, 384))

    img = torch.tensor(img.transpose((2,0,1))).unsqueeze(0).float()
    print(img.shape)

    pdb.set_trace()

    pred = yolo(img)
    
    bbox = non_max_suppression(pred[0][0])

    print(bbox)


if __name__ == '__main__':
    main()
