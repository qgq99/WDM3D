"""
-*- coding: utf-8 -*-
@Time    : 2024-11-09 20:46:07
@Author  : Qin Guoqing
@File    : temp.py
@Description : Description of this file
"""

from utils.wdm3d_utils import random_bbox2d


G = globals()


def main():
    for _ in range(100):
        print(random_bbox2d())


if __name__ == '__main__':
    main()
