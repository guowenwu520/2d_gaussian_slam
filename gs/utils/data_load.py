import csv
import glob
import os

import cv2
import numpy as np
import torch
import trimesh
from PIL import Image

class TUMParser:
    def __init__(self, input_folder):
        self.input_folder = input_folder
        # 加载位姿数据
        self.load_poses(self.input_folder, frame_rate=32)
        # 获取图像路径数量
        self.n_img = len(self.color_paths)

    def parse_list(self, filepath, skiprows=0):
        # 从文件中读取数据
        data = np.loadtxt(filepath, delimiter=" ", dtype=np.unicode_, skiprows=skiprows)
        return data

    def associate_frames(self, tstamp_image, tstamp_depth, tstamp_pose, max_dt=0.08):
            associations = []

            # 限制处理范围为索引 300 到 600
            start_idx = 0
            end_idx = len(tstamp_image)

            for i in range(start_idx, min(end_idx + 1, len(tstamp_image))):
                t = tstamp_image[i]

                if tstamp_pose is None:
                    j = np.argmin(np.abs(tstamp_depth - t))
                    if np.abs(tstamp_depth[j] - t) < max_dt:
                        associations.append((i, j))
                else:
                    j = np.argmin(np.abs(tstamp_depth - t))
                    k = np.argmin(np.abs(tstamp_pose - t))

                    if (np.abs(tstamp_depth[j] - t) < max_dt) and (
                        np.abs(tstamp_pose[k] - t) < max_dt
                    ):
                        associations.append((i, j, k))

            return associations

    def load_poses(self, datapath, frame_rate=-1):
        # 检查是否存在位姿数据
        if os.path.isfile(os.path.join(datapath, "groundtruth.txt")):
            pose_list = os.path.join(datapath, "groundtruth.txt")
        elif os.path.isfile(os.path.join(datapath, "pose.txt")):
            pose_list = os.path.join(datapath, "pose.txt")

        image_list = os.path.join(datapath, "rgb.txt")
        depth_list = os.path.join(datapath, "depth.txt")
        self.is_plane_info = os.path.isfile(os.path.join(datapath, "plane.txt"))
      
        # 读取图像、深度和位姿数据
        image_data = self.parse_list(image_list)
        depth_data = self.parse_list(depth_list)
        pose_data = self.parse_list(pose_list, skiprows=1)
        pose_vecs = pose_data[:, 0:].astype(np.float64)

        tstamp_image = image_data[:, 0].astype(np.float64)
        tstamp_depth = depth_data[:, 0].astype(np.float64)
        tstamp_pose = pose_data[:, 0].astype(np.float64)
        associations = self.associate_frames(tstamp_image, tstamp_depth, tstamp_pose)
            
        indicies = [0]
        for i in range(1, len(associations)):
            t0 = tstamp_image[associations[indicies[-1]][0]]
            t1 = tstamp_image[associations[i][0]]
            if t1 - t0 > 1.0 / frame_rate:
                indicies += [i]
        if self.is_plane_info:
            plane_list = os.path.join(datapath, "plane.txt") 
            plane_data = self.parse_list(plane_list)  
            # tstamp_plane = plane_data[:, 0].astype(np.float64)     

        self.color_paths, self.poses, self.depth_paths, self.frames ,self.plane_info = [], [], [], [],[]

        # 将图像、深度和位姿数据添加到列表中
        for ix in indicies:
            (i, j, k) = associations[ix]
            self.color_paths += [os.path.join(datapath, image_data[i, 1])]
            self.depth_paths += [os.path.join(datapath, depth_data[j, 1])]
            if self.is_plane_info:
                self.plane_info += [os.path.join(datapath, plane_data[i, 1])]

            quat = pose_vecs[k][4:]
            trans = pose_vecs[k][1:4]
            T = trimesh.transformations.quaternion_matrix(np.roll(quat, 1))
            T[:3, 3] = trans
            self.poses += [np.linalg.inv(T)]
            # print(datapath ,"  -- ",plane_data[i, 1])
            frame = {
                "file_path": str(os.path.join(datapath, image_data[i, 1])),
                "depth_path": str(os.path.join(datapath, depth_data[j, 1])),
                "plane_path": str(os.path.join(datapath, plane_data[i, 1])),
                "transform_matrix": (np.linalg.inv(T)).tolist(),
            }

            self.frames.append(frame)