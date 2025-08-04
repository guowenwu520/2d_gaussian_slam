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

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from utils.general_utils import PILtoTorch
import cv2
from PIL import Image
import torch
WARNED = False
class Camera(nn.Module):
    def load_image(self, use_mask=True):
        global WARNED
        image = Image.open(self.image_path)
        black_mask =None
        if self.depth_path != "":
            depth_rgb = cv2.imread(self.depth_path, cv2.IMREAD_COLOR)  # shape: (H, W, 3), BGR order
            if depth_rgb is None:
                raise FileNotFoundError(f"无法读取深度图像: {self.depth_path}")

            depth_rgb = cv2.cvtColor(depth_rgb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0  # 转为 RGB 并归一化

            # 创建 mask：黑色像素为遮罩（全0）
            black_mask = np.all(depth_rgb == 0.0, axis=-1).astype(np.float32) 
            try:
                if self.is_nerf_synthetic:
                    self.invdepthmap_2 = cv2.imread(self.depth_path, -1).astype(np.float32) / 512
                else:
                     self.invdepthmap_2 = cv2.imread(self.depth_path, -1).astype(np.float32) / float(2**16)
                if not WARNED:
                   print(f"read depth {self.depth_path}") 
            except FileNotFoundError:
                print(f"Error: The depth file at path '{self.depth_path}' was not found.")
                raise
            except IOError:
                print(f"Error: Unable to open the image file '{self.depth_path}'. It may be corrupted or an unsupported format.")
                raise
            except Exception as e:
                print(f"An unexpected error occurred when trying to read depth at {self.depth_path}: {e}")
                raise
        else:
            if not WARNED:
                print("not read depth") 
            self.invdepthmap_2 = None
        orig_w, orig_h = image.size

        if self.args.resolution in [1, 2, 4, 8]:
            resolution = round(orig_w/(self.resolution_scale * self.args.resolution)), round(orig_h/(self.resolution_scale * self.args.resolution))
        else:  # should be a type that converts to float
            if self.args.resolution == -1:
                if orig_w > 1600:
                    if not WARNED:
                        print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                            "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    global_down = orig_w / 1600
                else:
                    global_down = 1
            else:
                global_down = orig_w / self.args.resolution
        

            scale = float(global_down) * float(self.resolution_scale)
            resolution = (int(orig_w / scale), int(orig_h / scale))
        resized_image_rgb = PILtoTorch(image, resolution)
        gt_image = resized_image_rgb[:3, ...]

        self.mask_path = self.image_path.replace("/images/", "/masks/").replace(".jpg", ".png")

        if use_mask:
            try:
                mask_image = Image.open(self.mask_path).convert("L")
                mask_tensor = PILtoTorch(mask_image, resolution).clamp(0.0, 1.0)
                if not WARNED:
                   print(f"read mask {self.mask_path}") 
                if mask_tensor.dim() == 2:
                    mask_tensor = mask_tensor.unsqueeze(0)
            except Exception as e:
                print(f"Warning: Failed to load mask from {self.mask_path}, using default mask. Error: {e}")
                mask_tensor = None
        else:
            mask_tensor = None
        self.original_image = gt_image.clamp(0.0, 1.0).to(self.data_device)

        if mask_tensor is None:
            self.alpha_mask = None
            if resized_image_rgb.shape[0] == 4:
                self.alpha_mask = resized_image_rgb[3:4, ...].to(self.data_device)
            else: 
                self.alpha_mask = torch.ones_like(resized_image_rgb[0:1, ...].to(self.data_device))

            if self.train_test_exp and self.is_test_view:
                if is_test_dataset:
                    self.alpha_mask[..., :self.alpha_mask.shape[-1] // 2] = 0
                else:
                    self.alpha_mask[..., self.alpha_mask.shape[-1] // 2:] = 0
        else:
             self.alpha_mask = mask_tensor.to(self.data_device)

        self.invdepthmap = None
        self.depth_reliable = False
        if self.invdepthmap_2 is not None:
            self.depth_mask = torch.ones_like(self.original_image)
            self.invdepthmap = cv2.resize(self.invdepthmap_2, resolution)
            self.invdepthmap[self.invdepthmap < 0] = 0
            self.depth_reliable = True

            if self.depth_params is not None:
                if self.depth_params["scale"] < 0.2 * self.depth_params["med_scale"] or self.depth_params["scale"] > 5 * self.depth_params["med_scale"]:
                    self.depth_reliable = False
                    self.depth_mask *= 0
                
                if self.depth_params["scale"] > 0:
                    self.invdepthmap = self.invdepthmap * self.depth_params["scale"] + self.depth_params["offset"]

            if self.invdepthmap.ndim != 2:
                self.invdepthmap = self.invdepthmap[..., 0]
            self.invdepthmap = torch.from_numpy(self.invdepthmap[None]).to(self.data_device)
        # if self.invdepthmap is not None:     
        #     self.depth_mask *= self.invdepthmap  
        if black_mask is not None:
           self.alpha_mask = self.alpha_mask + torch.from_numpy(black_mask).to(self.data_device) 
        self.depth_mask = self.alpha_mask
        self.original_image *= (1.0 - self.alpha_mask)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]
        WARNED = True

    def unload_image(self):
        """释放已加载的图像和掩码，节省显存。"""
        del self.original_image
        self.original_image = None
        self.alpha_mask = None
        self.depth_mask = None
        self.invdepthmap_2 =None
        self.invdepthmap =None

        if hasattr(self, "alpha_mask"):
            del self.alpha_mask
            self.alpha_mask = None

        if hasattr(self, "image_width"):
            del self.image_width
            self.image_width = None

        if hasattr(self, "image_height"):
            del self.image_height
            self.image_height = None

        torch.cuda.empty_cache()  # 可选：清理 GPU 显存（谨慎使用）    


    def __init__(self, resolution, colmap_id, R, T, FoVx, FoVy, depth_params, invdepthmap,
                 image_name, uid,image_path=None,depth_path="",image=None,args=None,resolution_scale = 1.0,is_nerf_synthetic=True,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 train_test_exp = False, is_test_dataset = False, is_test_view = False
                 ):
        super(Camera, self).__init__()
        self.train_test_exp=train_test_exp
        self.is_test_dataset=is_test_dataset
        self.is_test_view=is_test_view
        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.depth_path = depth_path
        self.T = T
        self.FoVx = FoVx
        self.is_nerf_synthetic=is_nerf_synthetic
        self.FoVy = FoVy
        self.image_path = image_path
        self.image_name = image_name
        self.resolution = resolution
        self.depth_params =depth_params
        self.invdepthmap_2 =invdepthmap
        self.args=args
        self.resolution_scale =resolution_scale
        

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

       

        if image is not None:
                resized_image_rgb = PILtoTorch(image, resolution)
                gt_image = resized_image_rgb[:3, ...]
                self.alpha_mask = None
                if resized_image_rgb.shape[0] == 4:
                    self.alpha_mask = resized_image_rgb[3:4, ...].to(self.data_device)
                else: 
                    self.alpha_mask = torch.ones_like(resized_image_rgb[0:1, ...].to(self.data_device))

                if train_test_exp and is_test_view:
                    if is_test_dataset:
                        self.alpha_mask[..., :self.alpha_mask.shape[-1] // 2] = 0
                    else:
                        self.alpha_mask[..., self.alpha_mask.shape[-1] // 2:] = 0
                self.original_image = gt_image.clamp(0.0, 1.0).to(self.data_device)
                self.image_width = self.original_image.shape[2]
                self.image_height = self.original_image.shape[1]
                self.invdepthmap = None
                self.depth_reliable = False
                if invdepthmap is not None:
                    self.depth_mask = torch.ones_like(self.alpha_mask)
                    self.invdepthmap = cv2.resize(invdepthmap, resolution)
                    self.invdepthmap[self.invdepthmap < 0] = 0
                    self.depth_reliable = True

                    if depth_params is not None:
                        if depth_params["scale"] < 0.2 * depth_params["med_scale"] or depth_params["scale"] > 5 * depth_params["med_scale"]:
                            self.depth_reliable = False
                            self.depth_mask *= 0
                        
                        if depth_params["scale"] > 0:
                            self.invdepthmap = self.invdepthmap * depth_params["scale"] + depth_params["offset"]

                    if self.invdepthmap.ndim != 2:
                        self.invdepthmap = self.invdepthmap[..., 0]
                    self.invdepthmap = torch.from_numpy(self.invdepthmap[None]).to(self.data_device)
        else:
                self.original_image = None
                self.alpha_mask = None
        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        
class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

