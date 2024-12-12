"""
-*- coding: utf-8 -*-
@Time    : 2024-11-09 20:46:07
@Author  : Qin Guoqing
@File    : temp.py
@Description : Description of this file
"""

from utils.wdm3d_utils import load_config


G = globals()







def main():
    cfg = load_config("/home/qinguoqing/project/WDM3D/config/yolo/hyp.scratch-high.yaml", sub_cfg_keys=[])
    print(cfg)


if __name__ == '__main__':
    main()
