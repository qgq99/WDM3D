"""
-*- coding: utf-8 -*-
@Time    : 2024-11-13 12:30:49
@Author  : Qin Guoqing
@File    : wdm3d_utils.py.py
@Description : Description of this file
"""
import yaml


def create_module(g: dict, cfg: dict, module_name: str):
    """
    g: 应为globals()的返回值
    cfg: 配置对象
    """

    return g[cfg[module_name]["module"]](**cfg[module_name]["params"])


def load_config(config_path: str = "/home/qinguoqing/project/WDM3D/config/exp/exp.yaml", sub_cfg_keys=["dataset", "model"]):
    """
    加载配置对象, 包括数据配置、模型配置、损失配置等
    config_path:
    sub_cfg_keys: 需要加载的子配置文件
    """
    config = None
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    for k in sub_cfg_keys:
        config[k] = load_config(config[k], sub_cfg_keys=[])[k]
    
    return config
