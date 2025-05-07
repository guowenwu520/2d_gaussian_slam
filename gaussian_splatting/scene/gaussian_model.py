#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from torch import nn

from sklearn.neighbors import NearestNeighbors
from collections import deque
from gaussian_splatting.utils.general_utils import (
    build_rotation,
    build_scaling_rotation,
    get_expon_lr_func,
    helper,
    inverse_sigmoid,
    strip_symmetric,
)
from gaussian_splatting.utils.graphics_utils import BasicPointCloud, getWorld2View2
from gaussian_splatting.utils.sh_utils import RGB2SH
from gaussian_splatting.utils.system_utils import mkdir_p


class GaussianModel:
    def __init__(self, sh_degree: int, config=None):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree

        self._xyz = torch.empty(0, device="cuda")
        self._features_dc = torch.empty(0, device="cuda")
        self._features_rest = torch.empty(0, device="cuda")
        self._scaling = torch.empty(0, device="cuda")
        self._rotation = torch.empty(0, device="cuda")
        self._opacity = torch.empty(0, device="cuda")
        self.max_radii2D = torch.empty(0, device="cuda")
        self.xyz_gradient_accum = torch.empty(0, device="cuda")

        # -------------
        # self.spatial_lr_scale = 0
        self.unique_kfIDs = torch.empty(0).int()
        self.n_obs = torch.empty(0).int()
        self.optimizer = None
        # -------------

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = self.build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize
        # -------------
        self.config = config
        self.ply_input = None

        self.isotropic = False
        #----------------
    def gather_gaussians_to_cpu(self):
        return {
            "xyz": self._xyz,
            "f_dc": self._features_dc,
            "f_rest": self._features_rest,
            "scaling": self._scaling,
            "rotation": self._rotation,
            "opacity": self._opacity,
        }

    def build_covariance_from_scaling_rotation(
        self, scaling, scaling_modifier, rotation
    ):
        L = build_scaling_rotation(scaling_modifier * scaling, rotation)
        actual_covariance = L @ L.transpose(1, 2)
        symm = strip_symmetric(actual_covariance)
        return symm

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, self._rotation
        )

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_pcd_from_image(self, cam_info, init=False, scale=2.0, depthmap=None):
        cam = cam_info
        image_ab = (torch.exp(cam.exposure_a)) * cam.original_image + cam.exposure_b
        image_ab = torch.clamp(image_ab, 0.0, 1.0)
        rgb_raw = (image_ab * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy()

        if depthmap is not None:
            rgb = o3d.geometry.Image(rgb_raw.astype(np.uint8))
            depth = o3d.geometry.Image(depthmap.astype(np.float32))
        else:
            depth_raw = cam.depth
            if depth_raw is None:
                depth_raw = np.empty((cam.image_height, cam.image_width))

            if self.config["Dataset"]["sensor_type"] == "monocular":
                depth_raw = (
                    np.ones_like(depth_raw)
                    + (np.random.randn(depth_raw.shape[0], depth_raw.shape[1]) - 0.5)
                    * 0.05
                ) * scale

            rgb = o3d.geometry.Image(rgb_raw.astype(np.uint8))
            depth = o3d.geometry.Image(depth_raw.astype(np.float32))

        return self.create_pcd_from_image_and_depth(cam, rgb, depth, init)



    def region_growing(self,xyz, normals, radius=0.5, angle_threshold=np.deg2rad(15), min_cluster_size=400):
        """
        :param xyz: (N, 3) 点坐标
        :param normals: (N, 3) 点的单位法向量
        :param radius: 邻域半径
        :param angle_threshold: 法向夹角阈值（弧度）
        :param min_cluster_size: 最小平面点数
        :return: list of clusters，每个是索引数组
        """
        N = xyz.shape[0]
        visited = np.zeros(N, dtype=bool)
        clusters = []

        # 建立 KD-Tree 查找邻居
        nbrs = NearestNeighbors(radius=radius).fit(xyz)

        for idx in range(N):
            if visited[idx]:
                continue

            # 初始化一个新区域
            queue = deque([idx])
            visited[idx] = True
            cluster = [idx]

            while queue:
                curr_idx = queue.popleft()
                curr_normal = normals[curr_idx]
                _, neighbors = nbrs.radius_neighbors([xyz[curr_idx]], return_distance=True)

                for neighbor_idx in neighbors[0]:
                    if visited[neighbor_idx]:
                        continue
                    # 法向夹角小于阈值才合并
                    angle = np.arccos(np.clip(np.dot(curr_normal, normals[neighbor_idx]), -1.0, 1.0))
                    if angle < angle_threshold:
                        visited[neighbor_idx] = True
                        queue.append(neighbor_idx)
                        cluster.append(neighbor_idx)

            if len(cluster) >= min_cluster_size:
                clusters.append(np.array(cluster))
                break

        return clusters
    def fit_plane_svd(self,points):
        """
        对输入点集拟合平面，返回法向量和中心点
        :param points: (M, 3) array
        :return: normal: (3,), center: (3,)
        """
        center = points.mean(axis=0)
        _, _, vh = np.linalg.svd(points - center)
        normal = vh[2, :]
        # 方向统一化：确保法向量朝上（或者其他一致的方向）
        if normal[2] < 0:
            normal = -normal
        return normal, center
    def visualize_before_and_after(self,xyz, normals, adjusted_xyz, adjusted_normals):
        # 创建原始点云
        pcd_original = o3d.geometry.PointCloud()
        pcd_original.points = o3d.utility.Vector3dVector(xyz)
        pcd_original.normals = o3d.utility.Vector3dVector(normals)
        pcd_original.paint_uniform_color([0.7, 0.7, 0.7])  # 灰色

        # 创建调整后的点云
        pcd_adjusted = o3d.geometry.PointCloud()
        pcd_adjusted.points = o3d.utility.Vector3dVector(adjusted_xyz)
        pcd_adjusted.normals = o3d.utility.Vector3dVector(adjusted_normals)
        pcd_adjusted.paint_uniform_color([1.0, 0.0, 0.0])  # 绿色

        # 可视化
        o3d.visualization.draw_geometries([pcd_original, pcd_adjusted])

    def point_to_plane_distance(self,point, plane_center, plane_normal):
        """
        计算点到平面的距离，并返回调整后的点。
        :param point: 点的位置 (3,)
        :param plane_center: 平面的中心点 (3,)
        :param plane_normal: 平面的法向量 (3,)
        :return: 距离、调整后的点
        """
        # 计算点到平面的向量
        point_to_plane_vector = point - plane_center
        # 点到平面的距离（点到平面的投影长度）
        distance = np.dot(point_to_plane_vector, plane_normal)
        # 调整后的点
        adjusted_point = point - distance * plane_normal
        return distance, adjusted_point

    def adjust_points_to_planes(self,xyz, planes,normals, distance_threshold=0.1):
        """
        调整点到平面上，并更新每个点的法向量。
        :param xyz: 点坐标 (N, 3)
        :param planes: 包含平面信息的列表，每个平面包含 {"normal": normal, "center": center, "indices": indices}
        :param distance_threshold: 距离阈值，表示点到平面距离小于此值则进行调整
        :return: 更新后的点坐标和法向量
        """
        # 存储调整后的点坐标和法向量
        adjusted_xyz = np.copy(xyz)
        adjusted_normals = np.copy(normals)

        # 对于每个平面
        for plane in planes:
            normal = plane["normal"]
            center = plane["center"]
            cluster_indices = plane["indices"]

            # 对于平面上的每个点
            for idx in cluster_indices:
                point = xyz[idx]
                # 计算点到平面的距离
                distance, adjusted_point = self.point_to_plane_distance(point, center, normal)
              
                # 如果距离小于阈值，调整该点
                if np.abs(distance) < distance_threshold:
                    adjusted_xyz[idx] = adjusted_point
                    adjusted_normals[idx] = normal  # 更新法向量为平面的法向量

        return adjusted_xyz, adjusted_normals
    def transform_planes_to_world(self,plane_centers_cam, plane_normals_cam, R, T):
        """
        直接用 R, T 将相机坐标系下的平面中心和法向量转到世界坐标系。
        假设 R, T 是世界到相机（W2C）。

        Args:
            plane_centers_cam (torch.Tensor): [P, 3]，相机系下平面中心点
            plane_normals_cam (torch.Tensor): [P, 3]，相机系下平面法向量
            R (np.ndarray or torch.Tensor): [3, 3]，世界到相机的旋转矩阵
            T (np.ndarray or torch.Tensor): [3,]，世界到相机的平移向量

        Returns:
            plane_centers_world (torch.Tensor): [P, 3]
            plane_normals_world (torch.Tensor): [P, 3]
        """
        # if isinstance(R, np.ndarray):
        #     R = torch.from_numpy(R).to(dtype=torch.float32, device=plane_centers_cam.device)
        # else:
        #     R = R.to(dtype=torch.float32, device=plane_centers_cam.device)

        # if isinstance(T, np.ndarray):
        #     T = torch.from_numpy(T).to(dtype=torch.float32, device=plane_centers_cam.device)
        # else:
        #     T = T.to(dtype=torch.float32, device=plane_centers_cam.device)


        # # 世界到相机 -> 相机到世界： x_world = Rᵀ @ (x_cam - T)
        # R_t = R.transpose(0, 1)  # Rᵀ
        # centers_translated = plane_centers_cam - T  # [P, 3]
        # plane_centers_world = (R_t @ centers_translated.T).T

        # # 法向量只需旋转
        # plane_normals_world = (R_t @ plane_normals_cam.T).T
        # plane_normals_world = torch.nn.functional.normalize(plane_normals_world, dim=1)
        # return plane_centers_world, plane_normals_world
        x = plane_centers_cam[:, 0]
        y = plane_centers_cam[:, 1]
        z = plane_centers_cam[:, 2]
        transformed_centers = torch.stack([z, -y, x], dim=1)

        x = plane_normals_cam[:, 0]
        y = plane_normals_cam[:, 1]
        z = plane_normals_cam[:, 2]
        transformed_normals = torch.stack([z, -y, x], dim=1)
        return transformed_centers, transformed_normals
    def rotate_planes_to_z(self,centers, normals):
        # 目标法向量
        target = torch.tensor([0, 0, 1], dtype=torch.float32, device=normals.device).expand_as(normals)
        normals = torch.nn.functional.normalize(normals, dim=-1)

        # 旋转轴：法向量和目标向量的叉乘
        axis = torch.cross(normals, target, dim=1)
        axis_norm = torch.norm(axis, dim=1, keepdim=True)
        axis = axis / (axis_norm + 1e-8)

        # 角度
        cos = torch.clamp(torch.sum(normals * target, dim=1), -1.0, 1.0)
        angle = torch.acos(cos)

        # Rodrigues 旋转公式
        K = torch.zeros((centers.shape[0], 3, 3), device=normals.device)
        K[:, 0, 1] = -axis[:, 2]
        K[:, 0, 2] = axis[:, 1]
        K[:, 1, 0] = axis[:, 2]
        K[:, 1, 2] = -axis[:, 0]
        K[:, 2, 0] = -axis[:, 1]
        K[:, 2, 1] = axis[:, 0]

        I = torch.eye(3, device=normals.device).expand(centers.shape[0], 3, 3)
        angle = angle.view(-1, 1, 1)
        R = I + torch.sin(angle) * K + (1 - torch.cos(angle)) * (K @ K)

        # 旋转所有平面中心
        centers_rot = torch.bmm(R, centers.unsqueeze(-1)).squeeze(-1)
        normals_rot = torch.bmm(R, normals.unsqueeze(-1)).squeeze(-1)

        return centers_rot, normals_rot

    def create_pcd_from_image_and_depth(self, cam, rgb, depth, init=False):
            if init:
                downsample_factor = self.config["Dataset"]["pcd_downsample_init"]
            else:
                downsample_factor = self.config["Dataset"]["pcd_downsample"]
            point_size = self.config["Dataset"]["point_size"]
            if "adaptive_pointsize" in self.config["Dataset"]:
                if self.config["Dataset"]["adaptive_pointsize"]:
                    point_size = min(0.05, point_size * np.median(depth))
            
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                rgb,
                depth,
                depth_scale=1.0,
                depth_trunc=100.0,
                convert_rgb_to_intensity=False,
            )

            W2C = getWorld2View2(cam.R, cam.T).cpu().numpy()
            # print("W2C" ,rgbd.shape)
            pcd_tmp = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd,
                o3d.camera.PinholeCameraIntrinsic(
                    cam.image_width,
                    cam.image_height,
                    cam.fx,
                    cam.fy,
                    cam.cx,
                    cam.cy,
                ),
                extrinsic=W2C,
                project_valid_depth_only=True,
            )
            pcd_tmp = pcd_tmp.random_down_sample(1.0 / downsample_factor)

            pcd_tmp.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=1500))
            pcd_tmp.normalize_normals()

            xyz = np.asarray(pcd_tmp.points)
            rgb_np = np.asarray(pcd_tmp.colors)
            normals = np.asarray(pcd_tmp.normals)

            # plane_centers = torch.stack([torch.tensor(p['center'], device='cuda') for p in cam.plane_info], dim=0)  # [P, 3]
            # plane_normals = torch.stack([torch.tensor(p['normal'], device='cuda') for p in cam.plane_info], dim=0)  # [P, 3]
            # plane_colors = torch.stack([torch.tensor(p['color'], device='cuda') for p in cam.plane_info],dim=0).float() / 255.0  # [P, 3]
            
            # # plane_centers_cam, plane_normals_cam = self.rotate_planes_to_z(plane_centers, plane_normals)
            # plane_centers, plane_normals = self.transform_planes_to_world(plane_centers, plane_normals, cam.R, cam.T)
            # coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
            # # 将点云转换为 Open3D 格式
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(xyz)
            # pcd.colors = o3d.utility.Vector3dVector(rgb_np)
            # pcd.normals = o3d.utility.Vector3dVector(normals)
            # # 创建一个 Open3D 可视化器
            # vis = o3d.visualization.Visualizer()
            # vis.create_window()

            # vis.add_geometry(coord_frame)
            # # 添加点云到可视化器
            # vis.add_geometry(pcd)

            # # 创建平面并添加到可视化器
            # for i in range(len(plane_centers)):
            #     # 平面中心和法向量
            #     plane_center = plane_centers[i].cpu().numpy()
            #     plane_normal = plane_normals[i].cpu().numpy()

            # # 计算平面到原点的距离（点到平面的距离公式）
            #     plane_distance_to_origin = np.abs(np.dot(plane_center, plane_normal)) / np.linalg.norm(plane_normal)
            #     print(f"平面 {i+1} 到原点的距离: {plane_distance_to_origin:.4f} {plane_center} {plane_normal}")
            #     # 定义平面的四个顶点
            #     d = 1.0  # 平面边长
            #     half_d = d / 2.0

            #     # 四个顶点，围绕平面中心
            #     vertices = np.array([
            #         [half_d, half_d, 0],  # 顶点1
            #         [half_d, -half_d, 0], # 顶点2
            #         [-half_d, -half_d, 0], # 顶点3
            #         [-half_d, half_d, 0]   # 顶点4
            #     ])

            #     # 创建面，定义每个面由三个顶点组成
            #     faces = np.array([
            #         [0, 1, 2],
            #         [0, 2, 3]
            #     ])

            #     # 创建平面网格
            #     plane = o3d.geometry.TriangleMesh()

            #     # 设置平面顶点和面
            #     plane.vertices = o3d.utility.Vector3dVector(vertices)
            #     plane.triangles = o3d.utility.Vector3iVector(faces)

            #     # 根据法向量旋转平面
            #     rotation_matrix = o3d.geometry.get_rotation_matrix_from_xyz(np.arccos(plane_normal))
            #     plane.rotate(rotation_matrix, center=plane_center)

            #     # 平面平移到正确的位置
            #     plane.translate(plane_center) 
            #     print(plane_colors[i].cpu().numpy())
            #     plane.paint_uniform_color(plane_colors[i].cpu().numpy()) 

            #     # 添加平面到可视化器
            #     vis.add_geometry(plane)
            #      # 计算并检查点到原点的距离
            #     for j in range(len(xyz)):
            #         point = xyz[j]
            #         # 计算点到原点的距离
            #         point_distance_to_origin = np.linalg.norm(point)
            #         # print(f"点 {j+1} 到原点的距离: {point_distance_to_origin:.4f}")
              
            #         # 检查点到原点的距离是否等于平面到原点的距离
            #         if np.isclose(point_distance_to_origin, plane_distance_to_origin, atol=1e-3):  # 设定一个容忍误差
            #             print(f"点 {j+1} 到原点的距离 ({point_distance_to_origin:.4f}) 等于平面 {i+1} 到原点的距离")


            # # 启动可视化
            # vis.run()
        
            # === 保持原逻辑 ===
            pcd = BasicPointCloud(
                points=xyz, colors=rgb_np, normals=normals
            )
            self.ply_input = pcd

            fused_point_cloud = torch.from_numpy(np.asarray(pcd.points)).float().cuda()
            fused_color = RGB2SH(torch.from_numpy(np.asarray(pcd.colors)).float().cuda())
            features = (
                torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2))
                .float()
                .cuda()
            )
            features[:, :3, 0] = fused_color
            features[:, 3:, 1:] = 0.0

            dist2 = (
                torch.clamp_min(
                    distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()),
                    0.0000001,
                )
                * point_size
            )
            scales = torch.log(torch.sqrt(dist2))[..., None]
            if not self.isotropic:
                scales = scales.repeat(1, 3)

            rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
            rots[:, 0] = 1
            opacities = inverse_sigmoid(
                0.5
                * torch.ones(
                    (fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"
                )
            )

            return fused_point_cloud, features, scales, rots, opacities

    def init_lr(self, spatial_lr_scale):
        self.spatial_lr_scale = spatial_lr_scale

    def extend_from_pcd(
        self, fused_point_cloud, features, scales, rots, opacities, kf_id
    ):
        new_xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        new_features_dc = nn.Parameter(
            features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True)
        )
        new_features_rest = nn.Parameter(
            features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True)
        )
        new_scaling = nn.Parameter(scales.requires_grad_(True))
        new_rotation = nn.Parameter(rots.requires_grad_(True))
        new_opacity = nn.Parameter(opacities.requires_grad_(True))

        new_unique_kfIDs = torch.ones((new_xyz.shape[0])).int() * kf_id
        new_n_obs = torch.zeros((new_xyz.shape[0])).int()
        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            new_kf_ids=new_unique_kfIDs,
            new_n_obs=new_n_obs,
        )

    def extend_from_pcd_seq(
        self, cam_info, kf_id=-1, init=False, scale=2.0, depthmap=None
    ):
        fused_point_cloud, features, scales, rots, opacities = (
            self.create_pcd_from_image(cam_info, init, scale=scale, depthmap=depthmap)
        )
        self.extend_from_pcd(
            fused_point_cloud, features, scales, rots, opacities, kf_id
        )

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {
                "params": [self._xyz],
                "lr": training_args.position_lr_init * self.spatial_lr_scale,
                "name": "xyz",
            },
            {
                "params": [self._features_dc],
                "lr": training_args.feature_lr,
                "name": "f_dc",
            },
            {
                "params": [self._features_rest],
                "lr": training_args.feature_lr / 20.0,
                "name": "f_rest",
            },
            {
                "params": [self._opacity],
                "lr": training_args.opacity_lr,
                "name": "opacity",
            },
            {
                "params": [self._scaling],
                "lr": training_args.scaling_lr * self.spatial_lr_scale,
                "name": "scaling",
            },
            {
                "params": [self._rotation],
                "lr": training_args.rotation_lr,
                "name": "rotation",
            },
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps,
        )

        self.lr_init = training_args.position_lr_init * self.spatial_lr_scale
        self.lr_final = training_args.position_lr_final * self.spatial_lr_scale
        self.lr_delay_mult = training_args.position_lr_delay_mult
        self.max_steps = training_args.position_lr_max_steps

    def update_learning_rate(self, iteration):
        """Learning rate scheduling per step"""
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                # lr = self.xyz_scheduler_args(iteration)
                lr = helper(
                    iteration,
                    lr_init=self.lr_init,
                    lr_final=self.lr_final,
                    lr_delay_mult=self.lr_delay_mult,
                    max_steps=self.max_steps,
                )

                param_group["lr"] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append("f_dc_{}".format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append("f_rest_{}".format(i))
        l.append("opacity")
        for i in range(self._scaling.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self._rotation.shape[1]):
            l.append("rot_{}".format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = (
            self._features_dc.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        f_rest = (
            self._features_rest.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [
            (attribute, "f4") for attribute in self.construct_list_of_attributes()
        ]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            (xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1
        )
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.ones_like(self.get_opacity) * 0.01)
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def reset_opacity_nonvisible(
        self, visibility_filters
    ):  ##Reset opacity for only non-visible gaussians
        opacities_new = inverse_sigmoid(torch.ones_like(self.get_opacity) * 0.4)

        for filter in visibility_filters:
            opacities_new[filter] = self.get_opacity[filter]
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        def fetchPly_nocolor(path):
            plydata = PlyData.read(path)
            vertices = plydata["vertex"]
            positions = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T
            normals = np.vstack([vertices["nx"], vertices["ny"], vertices["nz"]]).T
            colors = np.ones_like(positions)
            return BasicPointCloud(points=positions, colors=colors, normals=normals)

        self.ply_input = fetchPly_nocolor(path)
        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("f_rest_")
        ]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape(
            (features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1)
        )

        scale_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("scale_")
        ]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("rot")
        ]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(
            torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._opacity = nn.Parameter(
            torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(
                True
            )
        )
        self._scaling = nn.Parameter(
            torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._rotation = nn.Parameter(
            torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self.active_sh_degree = self.max_sh_degree
        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")
        self.unique_kfIDs = torch.zeros((self._xyz.shape[0]))
        self.n_obs = torch.zeros((self._xyz.shape[0]), device="cpu").int()

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group["params"][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    (group["params"][0][mask].requires_grad_(True))
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    group["params"][0][mask].requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.unique_kfIDs = self.unique_kfIDs[valid_points_mask.cpu()]
        self.n_obs = self.n_obs[valid_points_mask.cpu()]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                    dim=0,
                )

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(
        self,
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_opacities,
        new_scaling,
        new_rotation,
        new_kf_ids=None,
        new_n_obs=None,
    ):
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        
        with torch.no_grad():  # 不追踪计算图，避免 autograd 报错
            log_scaling = self._scaling
            scaling = torch.exp(log_scaling)
            min_scaling_idx = torch.argmin(scaling, dim=1)

            new_log_scaling = log_scaling.clone()
            new_log_scaling[torch.arange(new_log_scaling.size(0)), min_scaling_idx] -= 0.001

            self._scaling.data.copy_(new_log_scaling)
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        if new_kf_ids is not None:
            self.unique_kfIDs = torch.cat((self.unique_kfIDs, new_kf_ids)).int()
        if new_n_obs is not None:
            self.n_obs = torch.cat((self.n_obs, new_n_obs)).int()

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[: grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values
            > self.percent_dense * scene_extent,
        )

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[
            selected_pts_mask
        ].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(
            self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)
        )
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        new_kf_id = self.unique_kfIDs[selected_pts_mask.cpu()].repeat(N)
        new_n_obs = self.n_obs[selected_pts_mask.cpu()].repeat(N)

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            new_kf_ids=new_kf_id,
            new_n_obs=new_n_obs,
        )

        prune_filter = torch.cat(
            (
                selected_pts_mask,
                torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool),
            )
        )

        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(
            torch.norm(grads, dim=-1) >= grad_threshold, True, False
        )
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values
            <= self.percent_dense * scene_extent,
        )

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_kf_id = self.unique_kfIDs[selected_pts_mask.cpu()]
        new_n_obs = self.n_obs[selected_pts_mask.cpu()]
        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
            new_kf_ids=new_kf_id,
            new_n_obs=new_n_obs,
        )

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent

            prune_mask = torch.logical_or(
                torch.logical_or(prune_mask, big_points_vs), big_points_ws
            )
        self.prune_points(prune_mask)

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True
        )
        self.denom[update_filter] += 1
