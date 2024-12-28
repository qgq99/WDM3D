"""
-*- coding: utf-8 -*-
@Time    : 2024-11-09 20:46:07
@Author  : Qin Guoqing
@File    : temp.py
@Description : Description of this file
"""

from utils.wdm3d_utils import load_config, create_module, Timer
from dataset.kitti.kitti import KITTIDataset
import torch
import os
from torch.utils.data import DataLoader

G = globals()



def infer_collate_fn(sample):
    sample = sample[0]
    batch_data = {}
    for k in sample.keys():
        if k == 'file_name':
            batch_data[k] = sample[k]
        else:
            batch_data[k] = torch.tensor(sample[k], device="cuda:0")
    return batch_data


def main():
    device = torch.device("cuda:0")
    # cfg = load_config("/home/qinguoqing/project/WDM3D/config/data/data.yaml", sub_cfg_keys=[])

if __name__ == '__main__':
    main()
