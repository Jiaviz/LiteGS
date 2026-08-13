#!/usr/bin/env python3
"""Project the official DTU laser scan into cameras as depth/normal supervision."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from plyfile import PlyData

from prepare_dtu import decompose_projection


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan-input", type=Path, required=True)
    parser.add_argument("--dataset-output", type=Path, required=True)
    parser.add_argument("--gt-ply", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=4, help="Image downsample factor for stored priors")
    parser.add_argument("--point-stride", type=int, default=2, help="Use every Nth official laser point")
    args = parser.parse_args()

    ply = PlyData.read(args.gt_ply).elements[0]
    points_original = np.stack([ply[name] for name in ("x", "y", "z")], axis=1).astype(np.float32)[::args.point_stride]
    normals_original = np.stack([ply[name] for name in ("nx", "ny", "nz")], axis=1).astype(np.float32)[::args.point_stride]
    archive = np.load(args.scan_input / "cameras.npz")
    scale_inv = archive["scale_mat_inv_0"].astype(np.float32)
    points_h = np.concatenate([points_original, np.ones((len(points_original), 1), np.float32)], axis=1)
    points_world = (points_h @ scale_inv.T)[:, :3]
    normal_linear = scale_inv[:3, :3]
    normals_world = normals_original @ np.linalg.inv(normal_linear)
    normals_world /= np.linalg.norm(normals_world, axis=1, keepdims=True).clip(1e-8)
    points_world = torch.from_numpy(points_world).cuda()
    normals_world = torch.from_numpy(normals_world).cuda()

    images = sorted(p for p in (args.scan_input / "image").glob("*.png") if not p.name.startswith("._"))
    depth_dir, normal_dir = args.dataset_output / "depths", args.dataset_output / "normals"
    depth_dir.mkdir(parents=True, exist_ok=True); normal_dir.mkdir(parents=True, exist_ok=True)
    for index, image_path in enumerate(images):
        projection = archive[f"world_mat_{index}"] @ archive[f"scale_mat_{index}"]
        k, rotation, translation = decompose_projection(projection)
        with Image.open(image_path) as image:
            width, height = image.size
        width //= args.resolution; height //= args.resolution
        k = k.copy(); k[:2] /= args.resolution
        rotation_t = torch.from_numpy(rotation.astype(np.float32)).cuda()
        translation_t = torch.from_numpy(translation.astype(np.float32)).cuda()
        camera = points_world @ rotation_t.T + translation_t
        uvw = camera @ torch.from_numpy(k.astype(np.float32)).cuda().T
        x = torch.round(uvw[:, 0] / uvw[:, 2]).long()
        y = torch.round(uvw[:, 1] / uvw[:, 2]).long()
        z = camera[:, 2]
        valid = (z > 0) & (x >= 0) & (x < width) & (y >= 0) & (y < height)
        ids = torch.nonzero(valid, as_tuple=False).flatten()
        pixel = y[ids] * width + x[ids]
        depth = torch.full((height * width,), float("inf"), device="cuda")
        depth.scatter_reduce_(0, pixel, z[ids], reduce="amin", include_self=True)
        # Select points within a thin z-buffer tolerance and average their normals.
        near = ids[z[ids] <= depth[pixel] + 5e-4]
        near_pixel = y[near] * width + x[near]
        view_normals = normals_world[near] @ rotation_t.T
        normal_sum = torch.zeros((height * width, 3), device="cuda")
        normal_sum.index_add_(0, near_pixel, view_normals)
        normal = torch.nn.functional.normalize(normal_sum, dim=1, eps=1e-6)
        depth[~torch.isfinite(depth)] = 0
        np.save(depth_dir / f"{image_path.stem}.npy", depth.reshape(height, width).cpu().numpy())
        np.save(normal_dir / f"{image_path.stem}.npy", normal.reshape(height, width, 3).cpu().numpy())
        print(f"[{index + 1}/{len(images)}] {image_path.name}: {(depth > 0).sum().item()} supervised pixels")


if __name__ == "__main__":
    main()
