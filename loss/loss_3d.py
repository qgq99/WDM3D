"""
-*- coding: utf-8 -*-
@Time    : 2024-12-12 14:31:45
@Author  : Qin Guoqing
@File    : loss_3d.py
@Description : loss computation for 3d prediction, code derived from WeakM3D.
"""

import torch
# from torch import nn
import torch.nn.functional as F
import numpy as np
# from utils.wdm3d_utils import create_module
# from torch.nn import MSELoss, SmoothL1Loss
# from utils.loss_tal_dual import ComputeLoss
import pdb


def calc_dis_ray_tracing(wl, Ry, points, density, bev_box_center):
    # pdb.set_trace()
    init_theta, length = torch.atan(
        wl[0] / wl[1]), torch.sqrt(wl[0] ** 2 + wl[1] ** 2) / 2  # 0.5:1
    corners = [((length * torch.cos(init_theta + Ry) + bev_box_center[0]).unsqueeze(0),
                (length * torch.sin(init_theta + Ry) + bev_box_center[1]).unsqueeze(0)),

               ((length * torch.cos(np.pi - init_theta + Ry) + bev_box_center[0]).unsqueeze(0),
                (length * torch.sin(np.pi - init_theta + Ry) + bev_box_center[1]).unsqueeze(0)),

               ((length * torch.cos(np.pi + init_theta + Ry) + bev_box_center[0]).unsqueeze(0),
                (length * torch.sin(np.pi + init_theta + Ry) + bev_box_center[1]).unsqueeze(0)),

               ((length * torch.cos(-init_theta + Ry) + bev_box_center[0]).unsqueeze(0),
                (length * torch.sin(-init_theta + Ry) + bev_box_center[1]).unsqueeze(0))]
    if Ry == np.pi/2:
        Ry -= 1e-4
    if Ry == 0:
        Ry += 1e-4
    k1, k2 = torch.tan(Ry), torch.tan(Ry + np.pi / 2)
    b11 = corners[0][1] - k1 * corners[0][0]
    b12 = corners[2][1] - k1 * corners[2][0]
    b21 = corners[0][1] - k2 * corners[0][0]
    b22 = corners[2][1] - k2 * corners[2][0]

    line0 = [k1, -1, b11]
    line1 = [k2, -1, b22]
    line2 = [k1, -1, b12]
    line3 = [k2, -1, b21]

    points[points[:, 0] == 0, 0] = 1e-4  # avoid inf
    #################################################
    slope_x = points[:, 1] / points[:, 0]
    intersect0 = torch.stack([line0[2] / (slope_x - line0[0]),
                              line0[2]*slope_x / (slope_x - line0[0])], dim=1)
    intersect1 = torch.stack([line1[2] / (slope_x - line1[0]),
                              line1[2]*slope_x / (slope_x - line1[0])], dim=1)
    intersect2 = torch.stack([line2[2] / (slope_x - line2[0]),
                              line2[2]*slope_x / (slope_x - line2[0])], dim=1)
    intersect3 = torch.stack([line3[2] / (slope_x - line3[0]),
                              line3[2]*slope_x / (slope_x - line3[0])], dim=1)

    dis0 = torch.abs(intersect0[:, 0] - points[:, 0]) + \
        torch.abs(intersect0[:, 1] - points[:, 1])
    dis1 = torch.abs(intersect1[:, 0] - points[:, 0]) + \
        torch.abs(intersect1[:, 1] - points[:, 1])
    dis2 = torch.abs(intersect2[:, 0] - points[:, 0]) + \
        torch.abs(intersect2[:, 1] - points[:, 1])
    dis3 = torch.abs(intersect3[:, 0] - points[:, 0]) + \
        torch.abs(intersect3[:, 1] - points[:, 1])

    dis_inter2center0 = torch.sqrt(intersect0[:, 0]**2 + intersect0[:, 1]**2)
    dis_inter2center1 = torch.sqrt(intersect1[:, 0]**2 + intersect1[:, 1]**2)
    dis_inter2center2 = torch.sqrt(intersect2[:, 0]**2 + intersect2[:, 1]**2)
    dis_inter2center3 = torch.sqrt(intersect3[:, 0]**2 + intersect3[:, 1]**2)

    intersect0 = torch.round(intersect0*1e4)
    intersect1 = torch.round(intersect1*1e4)
    intersect2 = torch.round(intersect2*1e4)
    intersect3 = torch.round(intersect3*1e4)

    dis0_in_box_edge = ((intersect0[:, 0] > torch.round(min(corners[0][0], corners[1][0])*1e4)) &
                        (intersect0[:, 0] < torch.round(max(corners[0][0], corners[1][0])*1e4))) | \
                       ((intersect0[:, 1] > torch.round(min(corners[0][1], corners[1][1])*1e4)) &
                        (intersect0[:, 1] < torch.round(max(corners[0][1], corners[1][1])*1e4)))
    dis1_in_box_edge = ((intersect1[:, 0] > torch.round(min(corners[1][0], corners[2][0])*1e4)) &
                        (intersect1[:, 0] < torch.round(max(corners[1][0], corners[2][0])*1e4))) | \
                       ((intersect1[:, 1] > torch.round(min(corners[1][1], corners[2][1])*1e4)) &
                        (intersect1[:, 1] < torch.round(max(corners[1][1], corners[2][1])*1e4)))
    dis2_in_box_edge = ((intersect2[:, 0] > torch.round(min(corners[2][0], corners[3][0])*1e4)) &
                        (intersect2[:, 0] < torch.round(max(corners[2][0], corners[3][0])*1e4))) | \
                       ((intersect2[:, 1] > torch.round(min(corners[2][1], corners[3][1])*1e4)) &
                        (intersect2[:, 1] < torch.round(max(corners[2][1], corners[3][1])*1e4)))
    dis3_in_box_edge = ((intersect3[:, 0] > torch.round(min(corners[3][0], corners[0][0])*1e4)) &
                        (intersect3[:, 0] < torch.round(max(corners[3][0], corners[0][0])*1e4))) | \
                       ((intersect3[:, 1] > torch.round(min(corners[3][1], corners[0][1])*1e4)) &
                        (intersect3[:, 1] < torch.round(max(corners[3][1], corners[0][1])*1e4)))

    dis_in_mul = torch.stack([dis0_in_box_edge, dis1_in_box_edge,
                              dis2_in_box_edge, dis3_in_box_edge], dim=1)
    dis_inter2cen = torch.stack([dis_inter2center0, dis_inter2center1,
                                 dis_inter2center2, dis_inter2center3], dim=1)
    dis_all = torch.stack([dis0, dis1, dis2, dis3], dim=1)

    # dis_in = torch.sum(dis_in_mul, dim=1).type(torch.bool)
    dis_in = (torch.sum(dis_in_mul, dim=1) == 2).type(torch.bool)
    if torch.sum(dis_in.int()) < 3:
        return 0
    """
    此处代码有逻辑问题
    dis_in由dis_in_mul降维计算得到, 直接用dis_in去索引必然报错

    from gpt:
    dis_in = (torch.sum(dis_in_mul, dim=1) == 2).type(torch.bool)
    这行代码检查每个交点是否满足同时位于两个边的范围内。这里使用 torch.sum(dis_in_mul, dim=1) 来计算每个交点与四个边的关系，
    只有当一个交点位于两个边之间时，其和才为 2。然后通过 == 2 得到一个布尔值，表示哪些交点满足这个条件。

    所以下面的索引操作应该是将位于两个边范围内的点选出来,
    dis_in_mul [1, 4, 100], 最后一维为点的索引, 4为4个边的索引, 所以将dis_inrepeat到[1, 4, 100]:
    dis_in = dis_in.repeat(4, 1).unsqueeze(0)
    """
    density_mask = dis_in.squeeze(0)
    dis_in = dis_in.repeat(4, 1).unsqueeze(0)

    dis_in_mul = dis_in_mul[dis_in]
    dis_inter2cen = dis_inter2cen[dis_in]
    dis_all = dis_all[dis_in]
    density = density[density_mask]

    z_buffer_ind = torch.argmin(dis_inter2cen[dis_in_mul].view(-1, 2), dim=1)
    z_buffer_ind_gather = torch.stack([~z_buffer_ind.byte(), z_buffer_ind.byte()],
                                      dim=1).type(torch.bool)
    # z_buffer_ind_gather = torch.stack([~(z_buffer_ind.type(torch.bool)), z_buffer_ind.type(torch.bool)],
    #                                   dim=1)

    # pdb.set_trace()
    dis = (dis_all[dis_in_mul].view(-1, 2)
           )[z_buffer_ind_gather] / density.unsqueeze(1)

    dis_mean = torch.mean(dis)
    return dis_mean


def calc_dis_rect_object_centric(wl, Ry, points, density):
    device = wl.device
    init_theta, length = torch.atan(
        wl[0] / wl[1]), torch.sqrt(wl[0] ** 2 + wl[1] ** 2) / 2  # 0.5:1
    corners = [((length * torch.cos(init_theta + Ry)).unsqueeze(0),
                (length * torch.sin(init_theta + Ry)).unsqueeze(0)),

               ((length * torch.cos(np.pi - init_theta + Ry)).unsqueeze(0),
                (length * torch.sin(np.pi - init_theta + Ry)).unsqueeze(0)),

               ((length * torch.cos(np.pi + init_theta + Ry)).unsqueeze(0),
                (length * torch.sin(np.pi + init_theta + Ry)).unsqueeze(0)),

               ((length * torch.cos(-init_theta + Ry)).unsqueeze(0),
                (length * torch.sin(-init_theta + Ry)).unsqueeze(0))]

    if Ry == np.pi/2:
        Ry -= 1e-4
    if Ry == 0:
        Ry += 1e-4

    k1, k2 = torch.tan(Ry), torch.tan(Ry + np.pi / 2)
    b11 = corners[0][1] - k1 * corners[0][0]
    b12 = corners[2][1] - k1 * corners[2][0]
    b21 = corners[0][1] - k2 * corners[0][0]
    b22 = corners[2][1] - k2 * corners[2][0]

    # line0 = [k1, -1, b11]
    # line1 = [k1, -1, b12]
    # line2 = [k2, -1, b21]
    # line3 = [k2, -1, b22]

    line0 = [k1, -1, b11]
    line1 = [k2, -1, b22]
    line2 = [k1, -1, b12]
    line3 = [k2, -1, b21]

    points[points[:, 0] == 0, 0] = 1e-4
    #################################################
    slope_x = points[:, 1] / points[:, 0]
    intersect0 = torch.stack([line0[2] / (slope_x - line0[0]),
                              line0[2]*slope_x / (slope_x - line0[0])], dim=1)
    intersect1 = torch.stack([line1[2] / (slope_x - line1[0]),
                              line1[2]*slope_x / (slope_x - line1[0])], dim=1)
    intersect2 = torch.stack([line2[2] / (slope_x - line2[0]),
                              line2[2]*slope_x / (slope_x - line2[0])], dim=1)
    intersect3 = torch.stack([line3[2] / (slope_x - line3[0]),
                              line3[2]*slope_x / (slope_x - line3[0])], dim=1)

    # dis0 = torch.sqrt((intersect0[:, 0] - points[:, 0])**2 +
    #                   (intersect0[:, 1] - points[:, 1])**2)
    # dis1 = torch.sqrt((intersect1[:, 0] - points[:, 0])**2 +
    #                   (intersect1[:, 1] - points[:, 1])**2)
    # dis2 = torch.sqrt((intersect2[:, 0] - points[:, 0])**2 +
    #                   (intersect2[:, 1] - points[:, 1])**2)
    # dis3 = torch.sqrt((intersect3[:, 0] - points[:, 0])**2 +
    #                   (intersect3[:, 1] - points[:, 1])**2)

    dis0 = torch.abs(intersect0[:, 0] - points[:, 0]) + \
        torch.abs(intersect0[:, 1] - points[:, 1])
    dis1 = torch.abs(intersect1[:, 0] - points[:, 0]) + \
        torch.abs(intersect1[:, 1] - points[:, 1])
    dis2 = torch.abs(intersect2[:, 0] - points[:, 0]) + \
        torch.abs(intersect2[:, 1] - points[:, 1])
    dis3 = torch.abs(intersect3[:, 0] - points[:, 0]) + \
        torch.abs(intersect3[:, 1] - points[:, 1])

    # dis_inter2center0 = torch.sqrt(intersect0[:, 0]**2 + intersect0[:, 1]**2)
    # dis_inter2center1 = torch.sqrt(intersect1[:, 0]**2 + intersect1[:, 1]**2)
    # dis_inter2center2 = torch.sqrt(intersect2[:, 0]**2 + intersect2[:, 1]**2)
    # dis_inter2center3 = torch.sqrt(intersect3[:, 0]**2 + intersect3[:, 1]**2)
    #
    # dis_point2center = torch.sqrt(points[:, 0]**2 + points[:, 1]**2)
    #################################################
    # pdb.set_trace()
    points_z = torch.cat([points, torch.zeros_like(points[:, :1])], dim=1)
    vec0 = torch.tensor([corners[0][0], corners[0][1], 0],
                        dtype=points_z.dtype, device=device)
    vec1 = torch.tensor([corners[1][0], corners[1][1], 0],
                        dtype=points_z.dtype, device=device)
    vec2 = torch.tensor([corners[2][0], corners[2][1], 0],
                        dtype=points_z.dtype, device=device)
    vec3 = torch.tensor([corners[3][0], corners[3][1], 0],
                        dtype=points_z.dtype, device=device)

    ''' calc direction of vectors'''
    cross0 = torch.cross(points_z, vec0.unsqueeze(
        0).repeat(points_z.shape[0], 1))[:, 2]
    cross1 = torch.cross(points_z, vec1.unsqueeze(
        0).repeat(points_z.shape[0], 1))[:, 2]
    cross2 = torch.cross(points_z, vec2.unsqueeze(
        0).repeat(points_z.shape[0], 1))[:, 2]
    cross3 = torch.cross(points_z, vec3.unsqueeze(
        0).repeat(points_z.shape[0], 1))[:, 2]

    ''' calc angle across vectors'''
    norm_p = torch.sqrt(points_z[:, 0] ** 2 + points_z[:, 1] ** 2)

    norm_d = torch.sqrt(corners[0][0] ** 2 + corners[0]
                        [1] ** 2).repeat(points_z.shape[0], 1).view(-1)
    norm = norm_p * norm_d

    dot_vec0 = torch.matmul(points_z, vec0.unsqueeze(
        0).repeat(points_z.shape[0], 1).t())[:, 0]
    dot_vec1 = torch.matmul(points_z, vec1.unsqueeze(
        0).repeat(points_z.shape[0], 1).t())[:, 0]
    dot_vec2 = torch.matmul(points_z, vec2.unsqueeze(
        0).repeat(points_z.shape[0], 1).t())[:, 0]
    dot_vec3 = torch.matmul(points_z, vec3.unsqueeze(
        0).repeat(points_z.shape[0], 1).t())[:, 0]

    angle0 = torch.acos(dot_vec0/(norm))
    angle1 = torch.acos(dot_vec1/(norm))
    angle2 = torch.acos(dot_vec2/(norm))
    angle3 = torch.acos(dot_vec3/(norm))

    angle_sum0 = (angle0 + angle1) < np.pi
    angle_sum1 = (angle1 + angle2) < np.pi
    angle_sum2 = (angle2 + angle3) < np.pi
    angle_sum3 = (angle3 + angle0) < np.pi

    cross_dot0 = (cross0 * cross1) < 0
    cross_dot1 = (cross1 * cross2) < 0
    cross_dot2 = (cross2 * cross3) < 0
    cross_dot3 = (cross3 * cross0) < 0

    cross_all = torch.stack(
        [cross_dot0, cross_dot1, cross_dot2, cross_dot3], dim=1)
    angle_sum_all = torch.stack(
        [angle_sum0, angle_sum1, angle_sum2, angle_sum3], dim=1)

    # pdb.set_trace()
    choose_ind = cross_all & angle_sum_all
    dis_all = torch.stack([dis0, dis1, dis2, dis3],
                          dim=1) / density.view(-1, 1, 1)
    choose_dis = dis_all[choose_ind]
    choose_dis[choose_dis != choose_dis] = 0

    return choose_dis


def calc_3d_loss(pred_3D, batch_RoI_points, batch_lidar_y_center,
                 batch_lidar_orient, batch_lidar_density, P2,
                 bbox2d, batch_dim):
    """
    calc loss of 3d prediction, the code is derived from WeakM3D.
    """
    # pdb.set_trace()
    device = P2.device
    all_loss, count = 0, 0
    # per_sample = batch_RoI_points.shape[1]

    num_instance = bbox2d.shape[0]
    # sample_points_cnt = len(batch_RoI_points)
    fx = P2[0, 0].view(-1, 1).repeat(1, num_instance).view(-1)
    fy = P2[1, 1].view(-1, 1).repeat(1, num_instance).view(-1)
    cx = P2[0, 2].view(-1, 1).repeat(1, num_instance).view(-1)
    cy = P2[1, 2].view(-1, 1).repeat(1, num_instance).view(-1)

    p_locXY, p_locZ, p_ortConf = pred_3D

    # bbox2d = bbox2d.view(-1, 4)
    # p_locXY = p_locXY.view(-1, 2)
    # p_locZ = p_locZ.view(-1, 1)
    # p_ortConf = p_ortConf.view(-1, 2)

    p_locXYZ = torch.cat([p_locXY, p_locZ], dim=1)

    # num_instance = bbox2d.shape[0]

    h, w, center_x, center_y = bbox2d[:, 3] - bbox2d[:, 1], \
        bbox2d[:, 2] - bbox2d[:, 0], \
        (bbox2d[:, 0] + bbox2d[:, 2])/2, \
        (bbox2d[:, 1] + bbox2d[:, 3])/2
    # pdb.set_trace()
    # print(p_locXYZ.shape, torch.stack([w, h], dim=1).shape, torch.stack([center_x, center_y], dim=1).shape, torch.stack([cx, cy], dim=1).shape, torch.stack([fx, fy], dim=1).shape)
    proj_box_center = ((F.sigmoid(p_locXYZ[:, :2])-0.5) * torch.stack([w, h], dim=1) + torch.stack(
        [center_x, center_y], dim=1) - torch.stack([cx, cy], dim=1)) / torch.stack([fx, fy], dim=1)
    box_center = torch.cat([proj_box_center, torch.ones(
        (proj_box_center.shape[0], 1)).cuda()], dim=1)
    location_3d = p_locXYZ[:, 2:3] * box_center

    alpha_ratio = F.normalize(p_ortConf, dim=1)
    Ry_pred = torch.atan2(alpha_ratio[:, 0], alpha_ratio[:, 1]) % np.pi

    # pdb.set_trace()
    Ry = torch.tensor(batch_lidar_orient, device=device)
    Alpha = torch.tensor(-batch_lidar_orient, device=device).squeeze(1)
    trans_Ry = (torch.atan2(location_3d[:, 0], location_3d[:, 2])).detach()
    Alpha = (Alpha - trans_Ry) % np.pi

    # pdb.set_trace()
    # batch_RoI_points = batch_RoI_points.view(num_instance, -1, 3)
    # batch_lidar_y_center = batch_lidar_y_center.view(num_instance)
    # batch_lidar_orient = batch_lidar_orient.view(num_instance)
    # batch_lidar_density = batch_lidar_density.view(num_instance, -1)
    # abs_dim = batch_dim.view(num_instance, 3)

    for i in range(len(p_locXYZ)):
        single_Ry = Ry[i]
        single_wl = batch_dim[i, 1:]
        single_loc = location_3d[i]
        single_Alpha = Alpha[i]
        single_Ry_pred = Ry_pred[i]

        single_depth_points = torch.tensor(batch_RoI_points[i], device=device)
        single_density = torch.tensor(batch_lidar_density[i], device=device)
        single_lidar_center_y = torch.tensor(
            batch_lidar_y_center[i], device=device)

        if single_loc[2] > 3:
            ray_tracing_loss = calc_dis_ray_tracing(single_wl, single_Ry, single_depth_points[:, [0, 2]], single_density,
                                                    (single_loc[0], single_loc[2]))
        else:
            ray_tracing_loss = 0

        shift_depth_points = torch.stack([single_depth_points[:, 0] - single_loc[0],
                                          single_depth_points[:, 2] - single_loc[2]], dim=1)
        # pdb.set_trace()
        dis_error = calc_dis_rect_object_centric(
            single_wl, single_Ry, shift_depth_points, single_density)
        dis_error = torch.mean(dis_error)

        # '''center_loss'''
        center_loss = torch.mean(torch.abs(shift_depth_points[:, 0]) / single_density) + \
            torch.mean(torch.abs(shift_depth_points[:, 1]) / single_density)
        center_yloss = F.smooth_l1_loss(single_loc[1], single_lidar_center_y)

        # ''' LiDAR 3D box orient loss'''
        # pdb.set_trace()
        orient_loss = -1 * torch.cos(single_Alpha - single_Ry_pred)

        all_loss += dis_error + 0.1 * center_loss + ray_tracing_loss + \
            center_yloss + orient_loss
        count += 1

    if count == 0:
        return 0

    all_loss = all_loss / count
    return all_loss


def generate_data_for_loss(RoI_points, bbox2d, sample_roi_points=100, dim_prior=[[0.8, 1.8, 0.8], [0.6, 1.8, 1.8], [1.6, 1.8, 4.]]):
    """
    sample_roi_points: 一个实例采多少点
    """
    # pdb.set_trace()
    batch_RoI_points = np.zeros(
        (bbox2d.shape[0], sample_roi_points, 3), dtype=np.float32)
    batch_lidar_y_center = np.zeros((bbox2d.shape[0], 1), dtype=np.float32)
    batch_lidar_orient = np.zeros((bbox2d.shape[0], 1), dtype=np.float32)
    # batch_lidar_density = np.zeros(
    #     (bbox2d.shape[0], sample_roi_points), dtype=np.float32)
    batch_lidar_density = []
    batch_dim = []

    for i in range(bbox2d.shape[0]):
        # pdb.set_trace()
        # sample_points_cnt = RoI_points[i].shape[0]
        # print(f"{i} {len(RoI_points)}")
        y_coor = RoI_points[i][:, 1]
        batch_lidar_y_center[i] = np.mean(y_coor)
        y_thesh = (np.max(y_coor) + np.min(y_coor)) / 2
        y_ind = RoI_points[i][:, 1] > y_thesh

        y_ind_points = RoI_points[i][y_ind]
        if y_ind_points.shape[0] < 10:
            y_ind_points = RoI_points[i]

        rand_ind = np.random.randint(
            0, y_ind_points.shape[0], sample_roi_points)
        depth_points_sample = y_ind_points[rand_ind]
        batch_RoI_points[i] = depth_points_sample
        depth_points_np_xz = depth_points_sample[:, [0, 2]]

        '''orient'''
        orient_set = [(i[1] - j[1]) / (i[0] - j[0]) for j in depth_points_np_xz
                      for i in depth_points_np_xz]
        orient_sort = np.array(sorted(np.array(orient_set).reshape(-1)))
        orient_sort = np.arctan(orient_sort[~np.isnan(orient_sort)])
        orient_sort_round = np.around(orient_sort, decimals=1)
        set_orenit = list(set(orient_sort_round))

        ind = np.argmax([np.sum(orient_sort_round == i) for i in set_orenit])
        orient = set_orenit[ind]
        if orient < 0:
            orient += np.pi

        if orient > np.pi / 2 + np.pi * 3 / 8:
            orient -= np.pi / 2
        if orient < np.pi / 8:
            orient += np.pi / 2

        if np.max(RoI_points[i][:, 0]) - np.min(RoI_points[i][:, 0]) > 4 and \
                (orient >= np.pi / 8 and orient <= np.pi / 2 + np.pi * 3 / 8):
            if orient < np.pi / 2:
                orient += np.pi / 2
            else:
                orient -= np.pi / 2
        batch_lidar_orient[i] = orient

        '''
        density
        
        下面的代码计算了每个点到其他点的距离是多少
        np.sum(p_dis < 0.04, axis=1): 计算了对于每个点, 距离它0.04以内的点的数量是多少
        '''
        p_dis = np.array([(i[0] - depth_points_sample[:, 0]) ** 2 + (i[2] - depth_points_sample[:, 2]) ** 2
                          for i in depth_points_sample])
        batch_lidar_density.append(np.sum(p_dis < 0.04, axis=1))

        '''
        dim
        when only use car by default, cls_info[i] is always 2
        '''
        # cls_dim_prior = dim_prior[cls_info[i]]
        cls_dim_prior = dim_prior[2]
        batch_dim.append(cls_dim_prior)
    batch_dim = torch.tensor(batch_dim)
    # pdb.set_trace()
    return dict(
        batch_RoI_points=batch_RoI_points,
        batch_lidar_y_center=batch_lidar_y_center,
        batch_lidar_orient=batch_lidar_orient,
        batch_lidar_density=np.array(batch_lidar_density),
        batch_dim=batch_dim
    )
