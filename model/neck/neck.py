"""
-*- coding: utf-8 -*-
@Time    : 2024-06-19 17:37:17
@Author  : Qin Guoqing
@File    : neck.py.py
@Description : Description of this file
"""

import torch
from torch import nn
# from model.neck.hahi import HAHIHeteroNeck, build_base_neck
from model.neck.pemask_neck import LightPEMASKNeck, build_pe_mask_neck
from model.neck.dynamicpe_neck import DynamicPENeckSOFT, build_dynamic_pe_neck

import torch.nn.functional as F
import yaml

import pdb

class ComposedNeck(nn.Module):
    """
    根据GEDepth的模型结构, 将三个neck的计算过程整合到一个模块
    """
    def __init__(
            self, 
            base_neck: nn.Module, 
            pe_mask_neck: LightPEMASKNeck, 
            dynamic_pe_neck: DynamicPENeckSOFT, 
            depth_scale:int = 200,
            scales_base_neck_used:list = [2,3,4,5]
            ):
        """
        scales_base_neck_used: base neck使用了哪些尺度特征进行运算. eg: backbone输出的是[c1, c2, c3, c4, c5, c6], 
            该属性默认值[2,3,4,5]表示使用下标2,3,4,5处的特征图, 即[c3, c4, c5, c6]
        """
        super().__init__()
        self.base_neck = base_neck
        self.pe_mask_neck = pe_mask_neck
        self.dynamic_pe_neck = dynamic_pe_neck
        self.depth_scale = depth_scale
        self.scales_base_neck_used = scales_base_neck_used

        """
        indices: acoording to the depiction of GEDepth:
        "In line with the ground slope distribution in training data, we use 11 discrete angles evenly distributed in [−5, 5]."
        """
        self.indices = torch.linspace(-5, 5, 11, device=torch.cuda.current_device()).reshape((1, 11, 1, 1))
    

    def dynamic_pe(self, x, y, pe_img_comput):
        """
        
        x: output of the basic neck
        y: output of the pe_mask neck
        pe_img_comput: 'guessing' the slope map of the img
        """
        pe_img_comput = pe_img_comput.unsqueeze(1)
        pe_slope_k_for_loss = self.dynamic_pe_neck(x)
        # pdb.set_trace()
        pe_slope_k_for_loss = F.interpolate(pe_slope_k_for_loss, size=[pe_img_comput.shape[2], pe_img_comput.shape[3]],
                                    mode='bilinear')
        
        pe_slope_k = F.softmax(pe_slope_k_for_loss, dim=1)
        pe_slope_k = torch.sum(pe_slope_k * self.indices, dim=1)
        pe_slope_k = pe_slope_k.unsqueeze(1)
        pe_slope_k = torch.tan(torch.deg2rad(pe_slope_k))

        h = 1.65 # basic ground height for KITTI

        a = -h / (pe_img_comput + 1e-8)
        # pdb.set_trace()
        pe_offset = -h / ((a - pe_slope_k) + 1e-8)
        pe_offset_mask = pe_offset.clone()
        pe_offset_mask[pe_offset_mask < 0] = 0
        pe_offset_mask[pe_offset_mask > self.depth_scale] = 0
        pe_offset_mask[pe_offset_mask > 0] = 1
        pe_mask = (pe_offset * pe_offset_mask) * y
        return pe_mask, pe_slope_k_for_loss



        


    def forward(self, x, h, w, pe_img_comput):
        """
        
        """
        ori_x = x
        x = self.base_neck(x)
        """
        将base neck输出和初始特征融合一次
        """
        for i, v in enumerate(self.scales_base_neck_used):
            ori_x[v] += x[i]
        # pdb.set_trace()
        y, dynamic_y = self.pe_mask_neck(ori_x)
        y = F.interpolate(y, size=[h, w], mode="bilinear")
        pe_mask, pe_slope_k_ori = self.dynamic_pe(ori_x, y, pe_img_comput)
        return x, y, pe_mask, pe_slope_k_ori







def build_composed_neck(config_filepath: str):
    assert isinstance(config_filepath, str)

    with open(config_filepath) as f:
        cfg = yaml.safe_load(f)

    return ComposedNeck(
        # base_neck=build_base_neck(cfg["base_neck"]),
        base_neck=nn.Identity(),
        pe_mask_neck=build_pe_mask_neck(cfg["pe_mask_neck"]),
        dynamic_pe_neck=build_dynamic_pe_neck(cfg["dynamic_pe_neck"]),
        depth_scale=cfg["depth_scale"],
        scales_base_neck_used=cfg["scales_base_neck_used"]
    )
    





if __name__ == "__main__":
    # pdb.set_trace()
    # device = torch.device("cuda")
    # neck = build_composed_neck(cfg.MODEL.NECK).to(device)
    # # base = dla34(False).to(device)
    # dataset = KITTIDataset(cfg, "/home/qinguoqing/dataset/kitti/training")

    # data_loader = make_data_loader(cfg, is_train=True)

    # data = None
    # for data in data_loader:
    #     x = data['images'].to(device)
    #     h = x.tensors.shape[2]
    #     w = x.tensors.shape[3]
    #     print(h, w)
    #     targets = data['targets']
    #     pe_img_comput = torch.stack([t.get_field("slope_map") for t in targets]).to(device)
    #     x = base(x.tensors)
    #     neck_output = neck(x, h, w, pe_img_comput)
    #     for item in neck_output:
    #         if isinstance(item, tuple):
    #             for _ in item:
    #                 print(_.shape)
    #         else:
    #             print(item.shape)
    #     break    
    pass
