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

from gaussian_splatting.gaussian_render_2d import render as Render2D
from gaussian_splatting.gaussian_render_3d import render as Render3D

use2D = False

def render(
    viewpoint_camera,
    pc,
    pipe,
    bg_color,
   scaling_modifier=1.0,
    override_color=None,
    mask=None,
):
    if use2D:
        return Render2D(viewpoint_camera, pc, pipe, bg_color, scaling_modifier, override_color, mask)
    else:
        return Render3D(viewpoint_camera, pc, pipe, bg_color, scaling_modifier, override_color, mask)

