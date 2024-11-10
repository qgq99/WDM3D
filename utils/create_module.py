"""
-*- coding: utf-8 -*-
@Time    : 2024-11-10 20:57:56
@Author  : Qin Guoqing
@File    : create_module.py.py
@Description : Description of this file
"""


def create_module(g: dict, cfg: dict, module_name: str):
    """
    g: 应为globals()的返回值
    cfg: 配置对象
    """

    return g[cfg[module_name]["module"]](**cfg[module_name]["params"])
