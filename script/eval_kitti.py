"""
-*- coding: utf-8 -*-
@Time    : 2024-12-28 14:38:41
@Author  : Qin Guoqing
@File    : eval_kitti.py
@Description : Description of this file
"""

from model.model import WDM3D
from utils.wdm3d_utils import create_module,load_config
from utils.eval_utils import post_3d
from dataset.kitti.kitti import KITTIDataset
import argparse
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import os
import numpy as np
from tqdm import tqdm
import pdb

G = globals()







def get_collate_fn(device):

    def infer_collate_fn(sample):
        sample = sample[0]
        batch_data = {}
        for k in sample.keys():
            if k == 'file_name':
                batch_data[k] = sample[k]
            else:
                batch_data[k] = torch.tensor(sample[k], device=device)
        return batch_data
    
    return infer_collate_fn



def get_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--title", default="", help="experiment name")
    parser.add_argument("--desc", default="",
                        help="description of this experiment")
    parser.add_argument(
        "--config", default="/home/qinguoqing/project/WDM3D/config/exp/eval.yaml")
    parser.add_argument("--epoch", default=10, type=int)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--CUDA_VISIBLE_DEVICES", default="0")
    parser.add_argument("--device", default="cuda:0")

    parser.add_argument("--gt_dir", default="/home/qinguoqing/dataset/kitti/train/label_2")

    parser.add_argument("--output_dir", default="/home/qinguoqing/project/WDM3D/output/eval/debug")
    return parser


def main(args):
    device = torch.device(args.device)

    cfg = load_config(args.config)
    save_dir_exp = args.output_dir
    # pdb.set_trace()
    cfg["dataset"]["params"]["split"] = "val"
    kitti_valset = create_module(G, cfg, "dataset")
    val_dataloader = DataLoader(kitti_valset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=4,
                                drop_last=False,
                                collate_fn=lambda x: x
                                )

    
    model = WDM3D(cfg["model"]).to(device)


    
    if not os.path.exists(save_dir_exp):
        os.mkdir(save_dir_exp)
    with torch.no_grad():
        model.eval()
        for batch_idx, sample in tqdm(enumerate(val_dataloader)):
            # batch_input = build_dataloader.process_batch_data(sample)
            pdb.set_trace()

            sample = sample[0]
            batch_input = {}
            for k in sample.keys():
                if k == 'file_name':
                    batch_input[k] = sample[k]
                else:
                    batch_input[k] = torch.tensor(sample[k], device=device)


            P2 = batch_input['P2'][0].cpu().numpy()
            bbox2d = batch_input['bbox2d'][0].cpu().numpy()
            det_2D = batch_input['det_2D'][0].cpu().numpy()
            file_name = batch_input['file_name'][0]

            if bbox2d.shape[0] < 1:
                np.savetxt('{}/{}.txt'.format(save_dir_exp, file_name), np.array([]), fmt='%s')
                continue

            pred_3D = model(batch_input['l_img'], batch_input['bbox2d'])

            p_locxy, p_locZ, p_ortConf = pred_3D
            p_locXYZ = torch.cat([p_locxy, p_locZ], dim=1)

            fx, fy, cx, cy = P2[0][0], P2[1][1], P2[0][2], P2[1][2]

            det_3D = np.zeros((p_locXYZ.shape[0], 16), dtype=object)
            det_3D[:, 0] = ['Car' for _ in range(p_locXYZ.shape[0])]
            det_3D[:, 4:8] = det_2D[:, 1:5]
            det_3D[:, -1] = det_2D[:, -1]
            '''car dimension'''
            det_3D[:, 8:11] = [np.array(cfg["loss"]["params"]["dim_prior"][2]) for _ in range(p_locXYZ.shape[0])]

            for i in range(len(p_locXYZ)):
                p, b = p_locXYZ[i], det_2D[i, 1:5]
                h, w, center_x, center_y = b[3] - b[1], b[2] - b[0], (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                proj_box_center = ((F.sigmoid(p[:2]) - 0.5) * torch.tensor([w, h]).cuda() + \
                                   torch.tensor([center_x, center_y]).cuda() - \
                                   torch.tensor([cx, cy]).cuda()) / torch.tensor([fx, fy]).cuda()
                proj_box_center = torch.cat([proj_box_center, torch.tensor([1.]).cuda()])
                location_3d = p[2] * proj_box_center
                det_3D[i, 11:14] = location_3d.cpu().numpy()

                alpha_ratio = F.normalize((p_ortConf[i].unsqueeze(0))).squeeze(0)
                estimated_theta = torch.atan2(alpha_ratio[0], alpha_ratio[1])
                det_3D[i, 3] = float(estimated_theta)

                det_3D[i, 12] += float(det_3D[i, 8]) / 2
                det_3D[i, -2] = det_3D[i, 3] + np.arctan2(det_3D[i, 11], det_3D[i, 13])

            det_3D[:, 1:] = np.around(det_3D[:, 1:].astype(np.float64), decimals=5)
            np.savetxt('{}/{}.txt'.format(save_dir_exp, file_name), det_3D, fmt='%s')
        post_3d(save_dir_exp, save_dir_exp)
        eval.eval_from_scrach(args.gt_dir, save_dir_exp, ap_mode=11)
        eval.eval_from_scrach(args.gt_dir, save_dir_exp, ap_mode=40)

if __name__ == '__main__':
    main(get_arg_parser().parse_args())