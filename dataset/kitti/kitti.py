import pdb
from .kitti_utils import Calibration, read_label, approx_proj_center
from dataset.augmentations import get_composed_augmentations
from structures.params_3d import ParamsList
from model.layer.heatmap_coder import (
    gaussian_radius,
    draw_umich_gaussian,
    draw_gaussian_1D,
    draw_ellip_gaussian,
    draw_umich_gaussian_2D,
)
from pathlib import Path
from .kitti_utils import Calibration
from wavedata.obj_detection import *
from torch.utils.data import Dataset
import os
import logging
from math import ceil

import cv2
from sklearn.cluster import Birch
import numpy as np
import torch
from torch.nn import functional as F
from torchvision.transforms import Compose, ToTensor
from PIL import Image, ImageFile
from utils.general import xyxy2xywh

ImageFile.LOAD_TRUNCATED_IMAGES = True

TYPE_ID_CONVERSION = {
    'Car': 0,
    'Pedestrian': 1,
    'Cyclist': 2,
    'Van': -4,
    'Truck': -4,
    'Person_sitting': -2,
    'Tram': -99,
    'Misc': -99,
    'DontCare': -1,
}


def find_k(depth_map, pe):
    """
    copy-paste from code of GEDepth
    """
    a = 1.65 / pe
    b = 1.65 / depth_map
    k = b + a
    return k



default_transform = Compose([
    ToTensor()
])



class KITTIDataset(Dataset):
    def __init__(self, transforms=default_transform, augment=True, *args, **config):
        super(KITTIDataset, self).__init__()
        self.root = Path(config["root"])
        self.split = config["split"]

        self.image_dir = self.root / self.split / "image_2"
        self.image_right_dir = self.root / self.split / "image_3"
        self.label_dir = self.root / self.split / "label_2"
        self.calib_dir = self.root / self.split / "calib"
        self.planes_dir = self.root / self.split / "planes"

        self.is_train = self.split == "train"
        self.transforms = transforms
        # self.imageset_txt = self.root / self.split / \
        #     "ImageSets" / "trainval.txt"
        
        self.imageset_txt = self.root / "train" / \
            "ImageSets" / f"{self.split}.txt"

        assert os.path.exists(
            self.imageset_txt), "ImageSets file not exist, dir = {}".format(self.imageset_txt)

        image_files = []
        for line in open(self.imageset_txt, "r"):
            base_name = line.replace("\n", "")
            image_name = base_name + ".png"
            image_files.append(image_name)

        self.image_files = image_files
        self.label_files = [i.replace(".png", ".txt")
                            for i in self.image_files]
        self.use_ground_plane = config["use_ground_plane"]

        self.horizon_gaussian_radius = config["horizon_gaussian_radius"]
        self.modify_ground_plane_d = config["modify_ground_plane_d"]
        self.use_edge_slope = config["use_edge_slope"]
        self.pred_ground_plane = config["pred_ground_plane"]
        if self.use_ground_plane or self.pred_ground_plane:
            self.planes_files = self.label_files

        self.classes = config["detect_classes"]

        self.class2Idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.num_classes = len(self.classes)
        self.num_samples = len(self.image_files)

        # whether to use right-view image
        self.use_right_img = config["use_right_image"] & self.is_train

        self.augmentation = get_composed_augmentations(config) if (
            self.is_train and augment) else None

        # input and output shapes
        self.input_width = config["input_width"]
        self.input_height = config["input_height"]
        self.down_ratio = config["down_ratio"]
        self.output_width = self.input_width // config["down_ratio"]
        self.output_height = self.input_height // config["down_ratio"]
        self.output_size = [self.output_width, self.output_height]

        # maximal length of extracted feature map when appling edge fusion
        self.max_edge_length = (self.output_width + self.output_height) * 2
        self.max_objs = config["max_objects"]

        # filter invalid annotations
        self.filter_annos = config["filter_anno_enable"]
        self.filter_params = config["filter_annos"]

        # handling truncation
        self.consider_outside_objs = config["consider_outside_objs"]
        # whether to use approximate representations for outside objects
        self.use_approx_center = config["use_approx_center"]
        # the type of approximate representations for outside objects
        self.proj_center_mode = config["approx_3d_center"]

        # for edge feature fusion
        self.enable_edge_fusion = config["enable_edge_fusion"]

        # True
        self.use_modify_keypoint_visible = config["keypoint_visible_modify"]

        PI = np.pi
        self.orientation_method = config["orientation"]
        self.multibin_size = config["orientation_bin_size"]
        # centers for multi-bin orientation
        self.alpha_centers = np.array([0, PI / 2, PI, - PI / 2])

        # use '2D' or '3D' center for heatmap prediction
        self.heatmap_center = config["heatmap_center"]
        self.adjust_edge_heatmap = config["adjust_boundary_heatmap"]  # True
        # radius / 2d box, 0.5
        self.edge_heatmap_ratio = config["heat_map_ratio"]

        self.logger = logging.getLogger("monocd.dataset")
        self.logger.info("Initializing KITTI {} set with {} files loaded.".format(
            self.split, self.num_samples))

        self.loss_keys = config["loss_names"]
        self.filter_more_strictly = config["filter_more_strictly"]
        self.filter_more_smoothly = config["filter_more_smoothly"]
        # to filter something unimportant
        # # strictly filter
        if self.filter_more_strictly:
            assert not self.filter_more_smoothly
            self.min_height = 25
            self.min_depth = 2
            self.max_depth = 65
            self.max_truncation = 0.5
            self.max_occlusion = 2
        # smoothly filter
        if self.filter_more_smoothly:
            assert not self.filter_more_strictly
            self.min_height = 20
            self.min_depth = 0
            self.max_depth = 65
            self.max_truncation = 0.7
            self.max_occlusion = 2
        
        self.depth_mode = config["depth_mode"]

        """
		self.mxmn_vals_map存储每个图像的深度值的最值, 键为图像名称, 值为[mx, mn]
		"""
        self.depth_mxmn_vals_map = self.load_depth_map_mxmn_vals()

    def __len__(self):
        if self.use_right_img:
            return self.num_samples * 2
        else:
            return self.num_samples

    def get_image(self, idx):
        img_filename = os.path.join(self.image_dir, self.image_files[idx])
        # img = Image.open(img_filename).convert('RGB')

        img = cv2.imread(str(img_filename))
        return img

    def get_right_image(self, idx):
        img_filename = os.path.join(
            self.image_right_dir, self.image_files[idx])
        img = Image.open(img_filename).convert('RGB')
        return img

    def get_calibration(self, idx, use_right_cam=False):
        calib_filename = os.path.join(self.calib_dir, self.label_files[idx])
        return Calibration(calib_filename, use_right_cam=use_right_cam)

    def get_label_objects(self, idx):
        if self.split != 'test':
            label_filename = os.path.join(
                self.label_dir, self.label_files[idx])

        return read_label(label_filename)

    def get_ground_planes(self, idx):
        if self.split != 'test':
            # idx = int(self.planes_files[idx].split('.')[0])
            img_filename = Path(self.image_files[idx]).with_suffix(".txt")
            ground_plane = get_road_plane(img_filename, self.planes_dir)

        return ground_plane

    def get_edge_utils(self, image_size, pad_size, down_ratio=4):
        img_w, img_h = image_size

        x_min, y_min = np.ceil(
            pad_size[0] / down_ratio), np.ceil(pad_size[1] / down_ratio)
        x_max, y_max = (
            pad_size[0] + img_w - 1) // down_ratio, (pad_size[1] + img_h - 1) // down_ratio

        step = 1
        # boundary idxs
        edge_indices = []

        # left
        y = torch.arange(y_min, y_max, step)
        x = torch.ones(len(y)) * x_min

        edge_indices_edge = torch.stack((x, y), dim=1)
        edge_indices_edge[:, 0] = torch.clamp(edge_indices_edge[:, 0], x_min)
        edge_indices_edge[:, 1] = torch.clamp(edge_indices_edge[:, 1], y_min)
        edge_indices_edge = torch.unique(edge_indices_edge, dim=0)
        edge_indices.append(edge_indices_edge)

        # bottom
        x = torch.arange(x_min, x_max, step)
        y = torch.ones(len(x)) * y_max

        edge_indices_edge = torch.stack((x, y), dim=1)
        edge_indices_edge[:, 0] = torch.clamp(edge_indices_edge[:, 0], x_min)
        edge_indices_edge[:, 1] = torch.clamp(edge_indices_edge[:, 1], y_min)
        edge_indices_edge = torch.unique(edge_indices_edge, dim=0)
        edge_indices.append(edge_indices_edge)

        # right
        y = torch.arange(y_max, y_min, -step)
        x = torch.ones(len(y)) * x_max

        edge_indices_edge = torch.stack((x, y), dim=1)
        edge_indices_edge[:, 0] = torch.clamp(edge_indices_edge[:, 0], x_min)
        edge_indices_edge[:, 1] = torch.clamp(edge_indices_edge[:, 1], y_min)
        edge_indices_edge = torch.unique(
            edge_indices_edge, dim=0).flip(dims=[0])
        edge_indices.append(edge_indices_edge)

        # top
        x = torch.arange(x_max, x_min - 1, -step)
        y = torch.ones(len(x)) * y_min

        edge_indices_edge = torch.stack((x, y), dim=1)
        edge_indices_edge[:, 0] = torch.clamp(edge_indices_edge[:, 0], x_min)
        edge_indices_edge[:, 1] = torch.clamp(edge_indices_edge[:, 1], y_min)
        edge_indices_edge = torch.unique(
            edge_indices_edge, dim=0).flip(dims=[0])
        edge_indices.append(edge_indices_edge)

        # concatenate
        edge_indices = torch.cat([index.long()
                                 for index in edge_indices], dim=0)

        return edge_indices

    def encode_alpha_multibin(self, alpha, num_bin=2, margin=1 / 6):
        # encode alpha (-PI ~ PI) to 2 classes and 1 regression
        encode_alpha = np.zeros(num_bin * 2)
        bin_size = 2 * np.pi / num_bin  # pi
        margin_size = bin_size * margin  # pi / 6

        bin_centers = self.alpha_centers
        range_size = bin_size / 2 + margin_size

        offsets = alpha - bin_centers
        offsets[offsets > np.pi] = offsets[offsets > np.pi] - 2 * np.pi
        offsets[offsets < -np.pi] = offsets[offsets < -np.pi] + 2 * np.pi

        for i in range(num_bin):
            offset = offsets[i]
            if abs(offset) < range_size:
                encode_alpha[i] = 1
                encode_alpha[i + num_bin] = offset

        return encode_alpha

    def filtrate_objects(self, obj_list):
        """
        Discard objects which are not in self.classes (or its similar classes)
        :param obj_list: list
        :return: list
        """
        type_whitelist = self.classes
        valid_obj_list = []
        for obj in obj_list:
            if obj.type not in type_whitelist:
                continue

            if self.filter_more_smoothly or self.filter_more_strictly:
                if (obj.occlusion > self.max_occlusion) or (obj.truncation > self.max_truncation) or ((obj.ymax - obj.ymin) < self.min_height) or (
                        obj.t[-1] > self.max_depth) or (obj.t[-1] < self.min_depth):
                    continue

            valid_obj_list.append(obj)

        return valid_obj_list

    def pad_image(self, image):
        # img = np.array(image)
        h, w, c = image.shape
        ret_img = np.zeros((self.input_height, self.input_width, c))
        pad_y = (self.input_height - h) // 2
        pad_x = (self.input_width - w) // 2

        ret_img[pad_y: pad_y + h, pad_x: pad_x + w] = image
        pad_size = np.array([pad_x, pad_y])

        return ret_img.astype(np.uint8), pad_size

    def get_vertical_edge(self, img):
        """

        Args:
                img:

        Returns:
                [Whether the vertical edge is valid, predicted horizontal slope]

        """
        if not self.use_edge_slope:
            """
            TODO
            pred_ground_plane为True时要调用该函数计算结果, 但又配置了use_edge_slope为False导致实际上没有计算
            不是是否为有意为之
            """
            return [False, -1]

        # GPEnet style
        Blur_img = cv2.GaussianBlur(img, ksize=(13, 13), sigmaX=4, sigmaY=4)
        CannyEdges = cv2.Canny(Blur_img, threshold1=50,
                               threshold2=100, apertureSize=3)
        HoughEdges = cv2.HoughLinesP(CannyEdges, rho=1, theta=np.pi / 180, threshold=5, minLineLength=40,
                                     maxLineGap=10)

        # other style
        # Blur_img = cv2.GaussianBlur(img, ksize=(13, 13), sigmaX=4, sigmaY=4)
        # Blur_img_gray = cv2.cvtColor(Blur_img, cv2.COLOR_RGB2GRAY)
        # lsd = cv2.createLineSegmentDetector(0)
        # HoughEdges = lsd.detect(Blur_img_gray)[0]

        VerticalEdgesAngles = []
        if HoughEdges is None:
            return [False, -1]
        for line in HoughEdges:
            x1, y1, x2, y2 = line[0]
            if np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) < 40:
                continue
            # transform slope to angle
            if x2 - x1 == 0:
                angle = 90
            else:
                slope = (y2 - y1) / (x2 - x1)
                angle = np.arctan(slope) * 180 / np.pi
            # limit the angle to 0~180
            if angle < 0:
                angle = angle + 180
            if 80 < angle < 100:
                VerticalEdgesAngles.append(angle)

        Nv = len(VerticalEdgesAngles)
        if Nv == 0:
            return [False, -1]
        Sv = np.std(VerticalEdgesAngles)
        # if Nv > 5 and Sv < 2:
        if Nv > 3 and Sv < 3:
            vertical_edges_angle = np.array(VerticalEdgesAngles).reshape(-1, 1)
            brc = Birch(n_clusters=None)
            brc.fit(vertical_edges_angle)
            labels = brc.predict(vertical_edges_angle)
            # find the cluster center of the most numerous class and use it as the slope of the ground plane
            unique, counts = np.unique(labels, return_counts=True)
            max_count_label = unique[np.argmax(counts)]
            ground_plane_angle = brc.subcluster_centers_[max_count_label][0]
            # transform angle to slope
            if ground_plane_angle == 90:
                Kh_pred = 0
            else:
                ground_plane_slope = np.tan(ground_plane_angle * np.pi / 180)
                Kh_pred = -1 / ground_plane_slope
            return [True, Kh_pred]
        else:
            return [False, -1]

    def generate_horizon_heat_map(self, horizon_heat_map, ground_plane, calib, pad_size, radius):
        a, b, c = ground_plane[0], ground_plane[1], ground_plane[2]
        f_x, f_y, c_x, c_y = calib.f_u, calib.f_v, calib.c_u, calib.c_v
        # compute the slope and intercept of the original image
        F = -b
        Kh = (a * f_y) / (f_x * F)
        bh = (c * f_y) / F + c_y - Kh * c_x
        # transform to downsampled image
        K = Kh
        B = (bh + pad_size[1] - Kh * pad_size[0]) / self.down_ratio

        # generate pixels on a horizontal line
        u = np.arange(0, horizon_heat_map.shape[2])
        v = K * u + B
        v = np.round(v).astype(int)
        for center in zip(u, v):
            horizon_heat_map[0] = draw_umich_gaussian(
                horizon_heat_map[0], center, radius)

        return horizon_heat_map

    def ploy_area(self, x, y):
        return 0.5*np.abs(np.dot(x, np.roll(y, 1))-np.dot(y, np.roll(x, 1)))

    def interp_corners3d(self, ground_corners3d, num_interp_x=100, num_interp_y=100, sampling_method='grid', sampling_num_type='fixed', ground_corners2d=None, down_ratio=4):
        assert len(ground_corners3d) == 4
        assert sampling_method in ['grid', 'random']
        assert sampling_num_type in ['fixed', 'area']
        if sampling_method == 'grid' and sampling_num_type == 'fixed':
            sampling_points = np.dstack(np.meshgrid(np.linspace(
                0, 1, num_interp_x), np.linspace(0, 1, num_interp_y))).reshape(-1, 2)
        elif sampling_method == 'random' and sampling_num_type == 'fixed':
            sampling_points = np.random.uniform(
                size=(num_interp_x * num_interp_y - 5, 2))
            sampling_points = np.concatenate([sampling_points, np.array(
                [[0.5, 0.5], [1., 1.], [0., 0.], [1., 0.], [0., 1.],])], axis=0)
        elif sampling_method == 'random' and sampling_num_type == 'area':
            area = self.ploy_area(
                ground_corners2d[:, 0], ground_corners2d[:, 1])
            area /= (down_ratio * down_ratio)
            if area > self.max_area:
                area = self.max_area  # 1600 * 928 / 32 / 32
            if area < 5:
                sampling_points = np.array(
                    [[0.5, 0.5], [1., 1.], [0., 0.], [1., 0.], [0., 1.],])
            else:
                sampling_points = np.random.uniform(size=(ceil(area) - 5, 2))
                sampling_points = np.concatenate([sampling_points, np.array(
                    [[0.5, 0.5], [1., 1.], [0., 0.], [1., 0.], [0., 1.],])], axis=0)
        else:
            raise NotImplementedError
        # if self.only_five_gdpts:
        # 	sampling_points = np.array([[0.5,0.5], [1.,1.], [0.,0.], [1.,0.], [0.,1.],])

        # num_interp_x * num_interp_y * 2
        x_vector = ground_corners3d[1] - ground_corners3d[0]
        z_vector = ground_corners3d[3] - ground_corners3d[0]

        base = np.stack([x_vector, z_vector])
        # vector from x and z directions

        sampled_pts = sampling_points @ base + ground_corners3d[0]

        return sampled_pts

    def calc_pe(self, h, w, calib: Calibration):
        """
        calculate the pre-computed ground embeding, the following calculation process is acoording to paper "GEDepth"
        h: height of the current image
        w: width of the current image
        calib: calibration of the current image
        """
        u, v = np.meshgrid(range(w), range(h), indexing="xy")

        P2, R0_rect, Tr_velo_to_cam = calib.P, calib.R0, calib.V2C

        K = P2[:, 0:3]
        R = R0_rect
        T = Tr_velo_to_cam[:, 3]
        A = np.linalg.inv(K @ R)
        B = np.linalg.inv(R) @ (- T)
        zc = (1.65 - B[1]) / (A[1, 0] * u + A[1, 1] * v + A[1, 2])

        return zc

    def load_depth_map_mxmn_vals(self):
        """
        加载将每个图像的深度图的实际深度最值, 该最值将用于把深度图的像素值还原为实际深度值
        """
        depth_mxmn_vals_map = {}
        path = self.root / self.split / "depth" / "max_min_val_map" / f"max_min_val_{self.depth_mode}.txt"
        with open(path) as f:
            for line in f:
                [img_name, mx, mn] = line.strip().split()
                depth_mxmn_vals_map[img_name] = [float(mx), float(mn)]
        return depth_mxmn_vals_map

    def load_depth_map(self, img_name):
        """
        加载由project/WDM3D/dataset/script/generate_depth_map.py生成的深度图
        img_name: 要加载的图像名称
        """

        path = self.root / self.split / "depth" / \
            f"depth_maps_{self.depth_mode}" / \
            f"{img_name}.png"
        [mx, mn] = self.depth_mxmn_vals_map[str(img_name)]

        depth = cv2.imread(path, -1)  # 不传第二个参数读取出来的将是3通道图
        depth = np.where(depth > 0, ((depth - 1) / 254) *
                         (mx - mn) + mn, depth)  # 归一化的逆操作

        if depth.shape != (self.input_height, self.input_width):
            # 若与输入尺寸不同则pad, 确保原图, 深度图以及pe尺寸一致
            ph, pw = self.input_height - depth.shape[0], self.input_width - depth.shape[1]
            depth = np.pad(depth, ((ph // 2, ph // 2 + int(ph % 2 == 1)),(pw // 2, pw // 2 + int(pw % 2 == 1))), "constant", constant_values=0)

        if self.depth_mode != "pred":   # mode为pred时, 不存在缺失深度值的像素
            depth[depth == 0] = np.inf  # 值为0表示缺失深度值, 暂用无穷大表示

        return depth

    def generate_pseudo_depth_map(self, h, w, locations, img_coords):
        """
        生成伪深度图
                - 伪深度图: 一个临时名称, 一个深度图, 只有每个目标中心处具有深度
                h, w: 图像尺寸
                locations: 各个目标中心点的坐标, 该坐标为相机坐标系坐标, z轴坐标即为深度
                img_coords: 各个目标中心点的像素坐标系坐标
        """
        img_coords = img_coords.astype(np.int16)  # 索引只能为整数
        pseudo_depth_map = np.zeros((h, w))
        for l, coord in zip(locations, img_coords):
            d = l[2]
            if 0 <= coord[0] < h and 0 <= coord[1] < w:
                # 由于截断, 个别目标的坐标点可能不再图像中，则无法在伪深度图中标注
                pseudo_depth_map[coord[0], coord[1]] = d
        return pseudo_depth_map

    def generate_slope_map(self, pe, depth_map):
        """
        !!!
        the code is from GEDepth, its meaning if not so clear,
        guessing the k(ie. k-img in the original code) refer to slope map mentioned in its paper.
        !!!
        """
        valid_mask = depth_map == 0
        k = find_k(depth_map, pe)
        k = np.around(np.rad2deg(np.arctan(k)))
        k[k > 5] = 5
        k[k < -5] = -5
        k[valid_mask] = 255

        return k

    def __getitem__(self, idx):

        if idx >= self.num_samples:
            # utilize right color image
            idx = idx % self.num_samples
            img = self.get_right_image(idx)
            calib = self.get_calibration(idx, use_right_cam=True)
            objs = None if self.split == 'test' else self.get_label_objects(
                idx)

            use_right_img = True
            # generate the bboxes for right color image
            right_objs = []
            img_w, img_h = img.size
            for obj in objs:
                corners_3d = obj.generate_corners3d()
                corners_2d, _ = calib.project_rect_to_image(corners_3d)
                obj.box2d = np.array([max(corners_2d[:, 0].min(), 0), max(corners_2d[:, 1].min(), 0),
                                      min(corners_2d[:, 0].max(), img_w - 1), min(corners_2d[:, 1].max(), img_h - 1)], dtype=np.float32)

                obj.xmin, obj.ymin, obj.xmax, obj.ymax = obj.box2d
                right_objs.append(obj)

            objs = right_objs
        else:
            # utilize left color image
            img = self.get_image(idx)
            calib = self.get_calibration(idx)
            objs = None if self.split == 'test' else self.get_label_objects(
                idx)
            ground_plane = None
            if self.use_ground_plane or self.pred_ground_plane:
                ground_plane = None if self.split == 'test' else self.get_ground_planes(
                    idx)
                if self.modify_ground_plane_d:
                    ground_plane[-1] = 1.65

            use_right_img = False

        original_idx = self.image_files[idx][:6]
        if not objs == None:
            # remove objects of irrelevant classes
            objs = self.filtrate_objects(objs)

        # random horizontal flip
        if self.augmentation is not None:
            img, objs, calib, ground_plane = self.augmentation(
                img, objs, calib, ground_plane=ground_plane)

        # pad image
        img_before_aug_pad = np.array(img).copy()
        if self.pred_ground_plane:
            horizon_state = self.get_vertical_edge(img_before_aug_pad)

        # img_w, img_h = img.size
        h, w, _ = img.shape
        img, pad_size = self.pad_image(img)
        # for training visualize, use the padded images
        ori_img = np.array(img).copy() if self.is_train else img_before_aug_pad

        # the boundaries of the image after padding and down_sampling
        x_min, y_min = int(np.ceil(
            pad_size[0] / self.down_ratio)), int(np.ceil(pad_size[1] / self.down_ratio))
        x_max, y_max = (
            pad_size[0] + w - 1) // self.down_ratio, (pad_size[1] + h - 1) // self.down_ratio

        if self.enable_edge_fusion:
            # generate edge_indices for the edge fusion module
            input_edge_indices = np.zeros(
                [self.max_edge_length, 2], dtype=np.int64)
            edge_indices = self.get_edge_utils(
                (w, h), pad_size).numpy()
            input_edge_count = edge_indices.shape[0]
            input_edge_indices[: edge_indices.shape[0]] = edge_indices
            input_edge_count = input_edge_count - 1  # explain ?

        if self.split == 'test':
            # for inference we parametrize with original size
            target = ParamsList(image_size=(h, w), is_train=self.is_train)
            target.add_field("pad_size", pad_size)
            target.add_field("calib", calib)
            target.add_field("ori_img", ori_img)
            if self.enable_edge_fusion:
                target.add_field('edge_len', input_edge_count)
                target.add_field('edge_indices', input_edge_indices)
            if self.pred_ground_plane:
                target.add_field('horizon_state', horizon_state)

            if self.transforms is not None:
                img, target = self.transforms(img, target)

            return img, target, original_idx

        # heatmap
        heat_map = np.zeros(
            [self.num_classes, self.output_height, self.output_width], dtype=np.float32)
        ellip_heat_map = np.zeros(
            [self.num_classes, self.output_height, self.output_width], dtype=np.float32)
        if self.pred_ground_plane:
            horizon_heat_map = np.zeros(
                [1, self.output_height, self.output_width], dtype=np.float32)
            horizon_heat_map = self.generate_horizon_heat_map(
                horizon_heat_map, ground_plane, calib, pad_size, self.horizon_gaussian_radius)

        # classification
        cls_ids = np.zeros([self.max_objs], dtype=np.int32)
        target_centers = np.zeros([self.max_objs, 2], dtype=np.int32)
        # 2d bounding boxes
        gt_bboxes = np.zeros([self.max_objs, 4], dtype=np.float32)
        bboxes = np.zeros([self.max_objs, 4], dtype=np.float32)
        # keypoints: 2d coordinates and visible(0/1)
        keypoints = np.zeros([self.max_objs, 10, 3], dtype=np.float32)
        # whether the depths computed from three groups of keypoints are valid
        keypoints_depth_mask = np.zeros([self.max_objs, 3], dtype=np.float32)

        ground_points = np.zeros([self.max_objs, 2], dtype=np.float32)

        # 3d dimension
        dimensions = np.zeros([self.max_objs, 3], dtype=np.float32)
        # 3d location
        locations = np.zeros([self.max_objs, 3], dtype=np.float32)
        # 3d EL
        EL = np.zeros([self.max_objs], dtype=np.float32)
        # rotation y
        rotys = np.zeros([self.max_objs], dtype=np.float32)
        # alpha (local orientation)
        alphas = np.zeros([self.max_objs], dtype=np.float32)
        # offsets from center to expected_center
        offset_3D = np.zeros([self.max_objs, 2], dtype=np.float32)

        # occlusion and truncation
        occlusions = np.zeros(self.max_objs)
        truncations = np.zeros(self.max_objs)

        if self.orientation_method == 'head-axis':
            orientations = np.zeros([self.max_objs, 3], dtype=np.float32)
        else:
            # multi-bin loss: 2 cls + 2 offset
            orientations = np.zeros(
                [self.max_objs, self.multibin_size * 2], dtype=np.float32)

        reg_mask = np.zeros([self.max_objs], dtype=np.uint8)  # regression mask
        # outside object mask
        trunc_mask = np.zeros([self.max_objs], dtype=np.uint8)
        # regression weight
        reg_weight = np.zeros([self.max_objs], dtype=np.float32)

        # print(f"obj cnt of image {idx}: {len(objs)}")
        
        obj_cnt = len(objs)
        # print(f"obj count of img {original_idx}: {obj_cnt}")
        bbox2d_gt = np.zeros((obj_cnt, 6))
        if obj_cnt > 0:
            # bbox2d_gt[:, 0] = idx
            bbox2d_gt[:, 1] = [self.class2Idx[i.type] for i in objs]
            bbox2d_gt[:, 2:] = xyxy2xywh(np.array([o.box2d for o in objs]))

        # print(bbox2d_gt)



        for i, obj in enumerate(objs):
            cls = obj.type
            cls_id = TYPE_ID_CONVERSION[cls]
            if cls_id < 0:
                continue

            # TYPE_ID_CONVERSION = {
            #     'Car': 0,
            #     'Pedestrian': 1,
            #     'Cyclist': 2,
            #     'Van': -4,
            #     'Truck': -4,
            #     'Person_sitting': -2,
            #     'Tram': -99,
            #     'Misc': -99,
            #     'DontCare': -1,
            # }

            # 0 for normal, 0.33 for partially, 0.66 for largely, 1 for unknown (mostly very far and small objs)
            float_occlusion = float(obj.occlusion)
            float_truncation = obj.truncation  # 0 ~ 1 and stands for truncation level

            # bottom centers ==> 3D centers
            locs = obj.t.copy()
            el = locs[1]
            """
			原本location为3D框底面中心在相机坐标系下的坐标, 下面一行将取了整个3D框中心的坐标作为location
			"""
            locs[1] = locs[1] - obj.h / 2
            if locs[-1] <= 0:
                continue  # objects which are behind the image

            # generate 8 corners of 3d bbox
            corners_3d = obj.generate_corners3d()
            corners_2d, _ = calib.project_rect_to_image(corners_3d)
            projected_box2d = np.array([corners_2d[:, 0].min(), corners_2d[:, 1].min(),
                                        corners_2d[:, 0].max(), corners_2d[:, 1].max()])

            if projected_box2d[0] >= 0 and projected_box2d[1] >= 0 and \
                    projected_box2d[2] <= w - 1 and projected_box2d[3] <= h - 1:
                box2d = projected_box2d.copy()
            else:
                box2d = obj.box2d.copy()

            # filter some unreasonable annotations
            if self.filter_annos:
                if float_truncation >= self.filter_params[0] and (box2d[2:] - box2d[:2]).min() <= self.filter_params[1]:
                    continue

            # project 3d location to the image plane
            proj_center, depth = calib.project_rect_to_image(
                locs.reshape(-1, 3))
            proj_center = proj_center[0]

            # generate approximate projected center when it is outside the image
            proj_inside_img = (
                0 <= proj_center[0] <= w - 1) & (0 <= proj_center[1] <= h - 1)

            approx_center = False
            if not proj_inside_img:
                if self.consider_outside_objs:
                    approx_center = True

                    center_2d = (box2d[:2] + box2d[2:]) / 2
                    if self.proj_center_mode == 'intersect':
                        target_proj_center, edge_index = approx_proj_center(
                            proj_center, center_2d.reshape(1, 2), (w, h))
                        if target_proj_center is None:
                            # print('Warning: center_2d is not in image')
                            continue
                    else:
                        raise NotImplementedError
                else:
                    continue
            else:
                target_proj_center = proj_center.copy()

            # 10 keypoints
            bot_top_centers = np.stack((corners_3d[:4].mean(
                axis=0), corners_3d[4:].mean(axis=0)), axis=0)
            keypoints_3D = np.concatenate(
                (corners_3d, bot_top_centers), axis=0)
            keypoints_2D, _ = calib.project_rect_to_image(keypoints_3D)

            # keypoints mask: keypoint must be inside the image and in front of the camera
            keypoints_x_visible = (keypoints_2D[:, 0] >= 0) & (
                keypoints_2D[:, 0] <= w - 1)
            keypoints_y_visible = (keypoints_2D[:, 1] >= 0) & (
                keypoints_2D[:, 1] <= h - 1)
            keypoints_z_visible = (keypoints_3D[:, -1] > 0)

            # xyz visible
            keypoints_visible = keypoints_x_visible & keypoints_y_visible & keypoints_z_visible
            # center, diag-02, diag-13
            keypoints_depth_valid = np.stack((keypoints_visible[[8, 9]].all(
            ), keypoints_visible[[0, 2, 4, 6]].all(), keypoints_visible[[1, 3, 5, 7]].all()))

            if self.use_modify_keypoint_visible:
                keypoints_visible = np.append(np.tile(keypoints_visible[:4] | keypoints_visible[4:8], 2), np.tile(
                    keypoints_visible[8] | keypoints_visible[9], 2))
                keypoints_depth_valid = np.stack((keypoints_visible[[8, 9]].all(
                ), keypoints_visible[[0, 2, 4, 6]].all(), keypoints_visible[[1, 3, 5, 7]].all()))

                keypoints_visible = keypoints_visible.astype(np.float32)
                keypoints_depth_valid = keypoints_depth_valid.astype(
                    np.float32)

            # downsample bboxes, points to the scale of the extracted feature map (stride = 4)
            keypoints_2D = (
                keypoints_2D + pad_size.reshape(1, 2)) / self.down_ratio
            target_proj_center = (target_proj_center +
                                  pad_size) / self.down_ratio
            proj_center = (proj_center + pad_size) / self.down_ratio

            box2d[0::2] += pad_size[0]
            box2d[1::2] += pad_size[1]
            box2d /= self.down_ratio
            # 2d bbox center and size
            bbox_center = (box2d[:2] + box2d[2:]) / 2
            bbox_dim = box2d[2:] - box2d[:2]

            # target_center: the point to represent the object in the downsampled feature map
            if self.heatmap_center == '2D':
                target_center = bbox_center.round().astype(np.int)
            else:
                target_center = target_proj_center.round().astype(int)

            # clip to the boundary
            target_center[0] = np.clip(target_center[0], x_min, x_max)
            target_center[1] = np.clip(target_center[1], y_min, y_max)

            pred_2D = True  # In fact, there are some wrong annotations where the target center is outside the box2d
            if not (target_center[0] >= box2d[0] and target_center[1] >= box2d[1] and target_center[0] <= box2d[2] and target_center[1] <= box2d[3]):
                pred_2D = False

            #
            if (bbox_dim > 0).all() and (0 <= target_center[0] <= self.output_width - 1) and (0 <= target_center[1] <= self.output_height - 1):
                rot_y = obj.ry
                alpha = obj.alpha

                # generating heatmap
                if self.adjust_edge_heatmap and approx_center:
                    # for outside objects, generate 1-dimensional heatmap
                    bbox_width = min(
                        target_center[0] - box2d[0], box2d[2] - target_center[0])
                    bbox_height = min(
                        target_center[1] - box2d[1], box2d[3] - target_center[1])
                    radius_x, radius_y = bbox_width * \
                        self.edge_heatmap_ratio, bbox_height * self.edge_heatmap_ratio
                    radius_x, radius_y = max(
                        0, int(radius_x)), max(0, int(radius_y))
                    assert min(radius_x, radius_y) == 0
                    heat_map[cls_id] = draw_umich_gaussian_2D(
                        heat_map[cls_id], target_center, radius_x, radius_y)
                else:
                    # for inside objects, generate circular heatmap
                    radius = gaussian_radius(bbox_dim[1], bbox_dim[0])
                    radius = max(0, int(radius))
                    heat_map[cls_id] = draw_umich_gaussian(
                        heat_map[cls_id], target_center, radius)

                cls_ids[i] = cls_id
                target_centers[i] = target_center
                # offset due to quantization for inside objects or offset from the interesection to the projected 3D center for outside objects
                offset_3D[i] = proj_center - target_center

                # 2D bboxes
                gt_bboxes[i] = obj.box2d.copy()  # for visualization
                if pred_2D:
                    bboxes[i] = box2d

                # local coordinates for keypoints
                keypoints[i] = np.concatenate(
                    (keypoints_2D - target_center.reshape(1, -1), keypoints_visible[:, np.newaxis]), axis=1)
                keypoints_depth_mask[i] = keypoints_depth_valid

                dimensions[i] = np.array([obj.l, obj.h, obj.w])
                locations[i] = locs
                rotys[i] = rot_y
                alphas[i] = alpha
                EL[i] = el
                orientations[i] = self.encode_alpha_multibin(
                    alpha, num_bin=self.multibin_size)

                reg_mask[i] = 1
                # all objects are of the same weights (for now)
                reg_weight[i] = 1
                # whether the center is truncated and therefore approximate
                trunc_mask[i] = int(approx_center)
                occlusions[i] = float_occlusion
                truncations[i] = float_truncation

        # visualization
        # img3 = show_image_with_boxes(img, cls_ids, target_centers, bboxes.copy(), keypoints, reg_mask,
        # 							offset_3D, self.down_ratio, pad_size, orientations, vis=True)

        # show_heatmap(img, heat_map, index=original_idx)
        # show_heatmap(img, horizon_heat_map, classes=['horizon'])

        depth_map = self.load_depth_map(
            Path(self.image_files[idx]).with_suffix(""))
        # calculate pre-computed ground embeding
        pe = self.calc_pe(self.input_height, self.input_width, calib)

        # if depth_map.shape != pe.shape:
        #     dh, dw = depth_map.shape
        #     """
        #     由于后续需要二者相加的操作, 因此若二者形状不同, 则将pe裁剪(depth_map形状为原图, pe经过transform更大)到与后者相同形状
        #     """
        #     pe = pe[:dh, :dw]

        # 每个目标中心在像素坐标系下的坐标
        # objs_center_image_coords, _ = calib.project_rect_to_image(locations)
        # pseudo_depth_map = self.generate_pseudo_depth_map(img.size[1], img.size[0], locations, objs_center_image_coords)
        slope_map = self.generate_slope_map(pe, depth_map)

        target = ParamsList(image_size=(h, w), is_train=self.is_train)
        target.add_field("cls_ids", cls_ids)
        target.add_field("target_centers", target_centers)
        target.add_field("keypoints", keypoints)
        target.add_field("keypoints_depth_mask", keypoints_depth_mask)
        target.add_field("dimensions", dimensions)
        target.add_field("locations", locations)
        target.add_field("EL", EL)
        target.add_field("calib", calib)

        if self.use_ground_plane or self.pred_ground_plane:
            target.add_field("ground_plane", ground_plane)
        if self.pred_ground_plane:
            target.add_field("horizon_heat_map", horizon_heat_map)
            target.add_field('horizon_state', horizon_state)
            target.add_field('horizon_vis_img', np.array(img))

        target.add_field("reg_mask", reg_mask)
        target.add_field("reg_weight", reg_weight)
        target.add_field("offset_3D", offset_3D)
        target.add_field("2d_bboxes", bboxes)
        target.add_field("pad_size", pad_size)
        target.add_field("ori_img", ori_img)
        target.add_field("rotys", rotys)
        target.add_field("trunc_mask", trunc_mask)
        target.add_field("alphas", alphas)
        target.add_field("orientations", orientations)
        target.add_field("hm", heat_map)
        # for validation visualization
        target.add_field("gt_bboxes", gt_bboxes)
        target.add_field("occlusions", occlusions)
        target.add_field("truncations", truncations)

        target.add_field("bbox2d_gt", bbox2d_gt)    # this is for yolov9 loss

        # set pre-computed ground embeding and slope map
        target.add_field("pe", pe)
        target.add_field("slope_map", slope_map)
        target.add_field("depth_map", depth_map)

        if self.enable_edge_fusion:
            target.add_field('edge_len', input_edge_count)
            target.add_field('edge_indices', input_edge_indices)

        if self.transforms is not None:
            # img, target = self.transforms(img, target)
            img = self.transforms(img)
        return img, target, original_idx
