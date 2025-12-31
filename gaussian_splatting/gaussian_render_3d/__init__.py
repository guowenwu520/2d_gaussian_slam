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

import math

from matplotlib import pyplot as plt

import torch
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)

from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.utils.sh_utils import eval_sh


def render(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
    mask=None,
):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    if pc.get_xyz.shape[0] == 0:
        return None

    screenspace_points = (
        torch.zeros_like(
            pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda"
        )
        + 0
    )
    try:
        screenspace_points.retain_grad()
    except Exception:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        projmatrix_raw=viewpoint_camera.projection_matrix,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    labels = pc.get_labels
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        # check if the covariance is isotropic
        if pc.get_scaling.shape[-1] == 1:
            scales = pc.get_scaling.repeat(1, 3)
        else:
            scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None

    #  # 确保标签是连续的0-based整数（若否，先映射）
    # unique_labels = labels.unique(sorted=True)
    # num_classes = len(unique_labels)
    # # print("num_classes: ", num_classes)
    # # 处理num_classes=1的边界情况（避免除以0）
    # if num_classes == 1:
    #     # 单类别时使用固定颜色（如红色），显式指定float32
    #     colors = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32, device=pc.get_features.device)
    # else:
    #     # 生成0到1的类别索引（CPU NumPy数组）
    #     class_indices = torch.arange(num_classes, device=pc.get_features.device).float()  # 默认是float32
    #     class_indices_cpu = (class_indices / (num_classes - 1)).cpu().numpy()  # 移至CPU并转为NumPy

    #     # 使用Matplotlib颜色映射（如'tab10'）
    #     cmap = plt.cm.get_cmap('tab10', num_classes)
    #     colors_np = cmap(class_indices_cpu)[:, :3]  # 获取RGB（形状：[num_classes, 3]）

    #     # 转回CUDA张量时显式指定float32类型
    #     colors = torch.tensor(colors_np, dtype=torch.float32, device=pc.get_features.device)

    # # 映射标签到颜色（假设标签是0-based连续整数）
    # fixed_color = colors[labels.long()]  # 确保labels是long类型索引
    # 替换原固定颜色为标签映射的颜色
    # fixed_color = point_colors
    # --------------------------

    if colors_precomp is None:
        if pipe.convert_SHs_python:
            # 若需使用 SHs 计算颜色（但我们要固定颜色，所以跳过此分支）
            shs_view = pc.get_features.transpose(1, 2).view(
                -1, 3, (pc.max_sh_degree + 1) ** 2
            )
            dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(
                pc.get_features.shape[0], 1
            )
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color  # 原始逻辑（若 override_color 存在）

    # --------------------------
    # 强制覆盖为固定颜色，并显式禁用 SHs
    # colors_precomp = fixed_color  # 使用预计算颜色
    # shs = None  # 关键！确保不传递 SHs

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    if mask is not None:
        rendered_image, radii, depth, opacity = rasterizer(
            means3D=means3D[mask],
            means2D=means2D[mask],
            shs=shs[mask],
            colors_precomp=colors_precomp[mask] if colors_precomp is not None else None,
            opacities=opacity[mask],
            scales=scales[mask],
            rotations=rotations[mask],
            cov3D_precomp=cov3D_precomp[mask] if cov3D_precomp is not None else None,
            theta=viewpoint_camera.cam_rot_delta,
            rho=viewpoint_camera.cam_trans_delta,
        )
    else:
        rendered_image, radii, depth, opacity, n_touched = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
            theta=viewpoint_camera.cam_rot_delta,
            rho=viewpoint_camera.cam_trans_delta,
        )
    print(f"n_touched: {n_touched.shape}")
    print(f"opacity: {opacity.shape}")
    print(f"depth: {depth.shape}")
    print(f"radii: {radii.shape}")
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "depth": depth,
        "opacity": opacity,
        "n_touched": n_touched,
        "3d_points": means3D,
        "labels":labels,
    }
