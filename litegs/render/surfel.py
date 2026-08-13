"""Perspective-correct CUDA 2D Gaussian surfel rendering for LiteGS."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
import torch.cuda.nvtx as nvtx

from .. import arguments
from ..utils import wrapper
from ..utils.statistic_helper import StatisticsHelper, StatisticsHelperInst


_NDC_GRID_CACHE: dict[tuple[torch.device, torch.dtype, int, int], tuple[torch.Tensor, torch.Tensor]] = {}


def _depth_to_normal(depth: torch.Tensor, proj_matrix: torch.Tensor) -> torch.Tensor:
    """Compute view-space normals from a camera-z depth image."""
    _, _, height, width = depth.shape
    fx_ndc = proj_matrix[0, 0]
    fy_ndc = proj_matrix[1, 1]
    cx_ndc = proj_matrix[2, 0]
    cy_ndc = proj_matrix[2, 1]
    cache_key = (depth.device, depth.dtype, height, width)
    cached_grid = _NDC_GRID_CACHE.get(cache_key)
    if cached_grid is None:
        yy, xx = torch.meshgrid(
            torch.arange(height, device=depth.device, dtype=depth.dtype),
            torch.arange(width, device=depth.device, dtype=depth.dtype),
            indexing="ij",
        )
        cached_grid = ((2.0 * xx + 1.0) / width - 1.0, (2.0 * yy + 1.0) / height - 1.0)
        _NDC_GRID_CACHE[cache_key] = cached_grid
    x_ndc, y_ndc = cached_grid
    z = depth[0, 0]
    points = torch.stack(((x_ndc - cx_ndc) * z / fx_ndc, (y_ndc - cy_ndc) * z / fy_ndc, z), dim=0)
    dx = points[:, 1:-1, 2:] - points[:, 1:-1, :-2]
    dy = points[:, 2:, 1:-1] - points[:, :-2, 1:-1]
    normals = F.normalize(torch.cross(dx, dy, dim=0), dim=0, eps=1e-6)
    return F.pad(normals, (1, 1, 1, 1))


def render(
    view_matrix: torch.Tensor,
    proj_matrix: torch.Tensor,
    xyz: torch.Tensor,
    scale: torch.Tensor,
    rot: torch.Tensor,
    sh_0: torch.Tensor,
    sh_rest: torch.Tensor,
    opacity: torch.Tensor,
    actived_sh_degree: int,
    output_shape: tuple[int, int],
    pp: arguments.PipelineParams,
    compute_surface_normal: bool = True,
):
    """Render RGB and geometric buffers with the official CUDA surfel kernel."""
    try:
        from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    except ImportError as error:
        raise ImportError(
            "2DGS needs the CUDA surfel extension. Install it with: "
            "pip install -v ./litegs/submodules/diff_surfel_rasterization"
        ) from error

    if view_matrix.shape[0] != 1 or proj_matrix.shape[0] != 1:
        raise ValueError("The 2DGS CUDA renderer currently expects one training view per call")

    height, width = output_shape
    means3d = xyz.transpose(0, 1).contiguous()
    scales = scale[:2].exp().transpose(0, 1).contiguous()
    rotations = F.normalize(rot, dim=0).transpose(0, 1).contiguous()
    opacities = opacity.sigmoid().transpose(0, 1).contiguous()
    means2d = torch.zeros_like(means3d, requires_grad=True)
    means2d.retain_grad()

    camera_center = (-view_matrix[..., 3:4, :3] @ view_matrix[..., :3, :3].transpose(-1, -2)).squeeze(1)
    directions = F.normalize(xyz - camera_center.unsqueeze(-1), dim=0)
    colors = wrapper.SphericalHarmonicToRGB.call_fused(actived_sh_degree, sh_0, sh_rest, directions)
    colors = colors[0].transpose(0, 1).contiguous()

    # LiteGS matrices use the same transposed (column-major) convention as 3DGS.
    full_proj = (view_matrix @ proj_matrix)[0].contiguous()
    tanfovx = float(1.0 / abs(proj_matrix[0, 0, 0].item()))
    tanfovy = float(1.0 / abs(proj_matrix[0, 1, 1].item()))
    settings = GaussianRasterizationSettings(
        image_height=height,
        image_width=width,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=torch.zeros(3, device=xyz.device, dtype=xyz.dtype),
        scale_modifier=1.0,
        viewmatrix=view_matrix[0].contiguous(),
        projmatrix=full_proj,
        sh_degree=0,
        campos=camera_center[0].contiguous(),
        prefiltered=False,
        debug=False,
    )
    nvtx.range_push("2DGS CUDA raster")
    image, radii, allmap = GaussianRasterizer(settings)(
        means3D=means3d,
        means2D=means2d,
        colors_precomp=colors,
        opacities=opacities,
        scales=scales,
        rotations=rotations,
    )
    nvtx.range_pop()

    alpha = allmap[1:2]
    rendered_normal = allmap[2:5]
    expected_depth = torch.nan_to_num(allmap[0:1] / alpha.clamp_min(1e-6))
    median_depth = torch.nan_to_num(allmap[5:6])
    surface_depth = expected_depth * (1.0 - pp.depth_ratio) + median_depth * pp.depth_ratio
    surface_normal = (
        _depth_to_normal(surface_depth.unsqueeze(0), proj_matrix[0]) * alpha.detach()
        if compute_surface_normal
        else None
    )
    distortion = allmap[6:7]

    if StatisticsHelperInst.bStart:
        visible = (radii > 0).unsqueeze(0)
        StatisticsHelperInst.update_visible_count(visible)
        StatisticsHelperInst.update_max_min("radii", radii.float().unsqueeze(0))
        StatisticsHelperInst.register_tensor_grad_callback(
            "mean2d_grad", means2d, StatisticsHelper.update_mean_std_compact,
            lambda tensor: tensor[:, :2].norm(dim=1),
        )
    return (
        image.unsqueeze(0),
        (1.0 - alpha).unsqueeze(0),
        surface_depth.unsqueeze(0),
        rendered_normal.unsqueeze(0),
        surface_normal.unsqueeze(0) if surface_normal is not None else None,
        distortion.unsqueeze(0),
        means2d,
        radii,
    )
