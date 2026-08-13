#!/usr/bin/env python3
"""Convert IDR-style DTU scans to a masked COLMAP/LiteGS dataset."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion
from scipy.spatial.transform import Rotation


def decompose_projection(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    intrinsics, rotation, center_h = cv2.decomposeProjectionMatrix(matrix[:3, :4])[:3]
    intrinsics /= intrinsics[2, 2]
    center = (center_h[:3] / center_h[3]).reshape(3)
    translation = -rotation @ center
    return intrinsics, rotation, translation


def visual_hull_points(
    cameras: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    masks: list[np.ndarray],
    bounds: tuple[float, float],
    resolution: int,
    min_view_ratio: float,
) -> np.ndarray:
    axis = np.linspace(bounds[0], bounds[1], resolution, dtype=np.float32)
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    votes = np.zeros(len(grid), dtype=np.uint16)
    visible = np.zeros(len(grid), dtype=np.uint16)
    for (k, rotation, translation), mask in zip(cameras, masks, strict=True):
        camera = grid @ rotation.T + translation
        valid_z = camera[:, 2] > 1e-6
        uvw = camera @ k.T
        uv = uvw[:, :2] / np.maximum(uvw[:, 2:3], 1e-6)
        x = np.rint(uv[:, 0]).astype(np.int32)
        y = np.rint(uv[:, 1]).astype(np.int32)
        inside = valid_z & (x >= 0) & (x < mask.shape[1]) & (y >= 0) & (y < mask.shape[0])
        visible += inside
        ids = np.flatnonzero(inside)
        votes[ids] += mask[y[ids], x[ids]] > 127
    keep = (visible >= max(3, len(cameras) // 3)) & (votes >= np.ceil(visible * min_view_ratio))
    # Initialize on the visual-hull boundary rather than filling the volume.
    # Both 3DGS and 2DGS therefore start from the same surface-biased geometry.
    volume = keep.reshape(resolution, resolution, resolution)
    boundary = volume & ~binary_erosion(volume, structure=np.ones((3, 3, 3), dtype=bool))
    points = grid[boundary.reshape(-1)]
    if len(points) == 0:
        raise RuntimeError("Visual hull is empty; check camera decomposition and masks")
    return points


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hull-resolution", type=int, default=96)
    parser.add_argument("--hull-bound", type=float, default=1.1)
    parser.add_argument("--min-view-ratio", type=float, default=0.9)
    args = parser.parse_args()

    images = sorted(p for p in (args.input / "image").glob("*.png") if not p.name.startswith("._"))
    masks = sorted(p for p in (args.input / "mask").glob("*.png") if not p.name.startswith("._"))
    if len(images) != len(masks):
        raise ValueError(f"Expected matching images/masks, got {len(images)} and {len(masks)}")
    archive = np.load(args.input / "cameras.npz")
    args.output.mkdir(parents=True, exist_ok=True)
    image_dir, mask_dir = args.output / "images", args.output / "masks"
    image_dir.mkdir(exist_ok=True); mask_dir.mkdir(exist_ok=True)

    camera_lines: list[str] = []
    image_lines: list[str] = []
    camera_data: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    loaded_masks: list[np.ndarray] = []
    for index, (image_path, mask_path) in enumerate(zip(images, masks, strict=True)):
        projection = archive[f"world_mat_{index}"] @ archive[f"scale_mat_{index}"]
        k, rotation, translation = decompose_projection(projection)
        with Image.open(image_path) as image:
            width, height = image.size
        camera_id = index + 1
        camera_lines.append(
            f"{camera_id} PINHOLE {width} {height} {k[0, 0]:.12g} {k[1, 1]:.12g} {k[0, 2]:.12g} {k[1, 2]:.12g}"
        )
        quat_xyzw = Rotation.from_matrix(rotation).as_quat()
        qx, qy, qz, qw = quat_xyzw
        image_lines.extend([
            f"{camera_id} {qw:.12g} {qx:.12g} {qy:.12g} {qz:.12g} "
            f"{translation[0]:.12g} {translation[1]:.12g} {translation[2]:.12g} {camera_id} {image_path.name}",
            "",
        ])
        shutil.copy2(image_path, image_dir / image_path.name)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        binary = (mask > 127).astype(np.uint8) * 255
        cv2.imwrite(str(mask_dir / image_path.name), binary)
        camera_data.append((k, rotation, translation))
        loaded_masks.append(binary)

    points = visual_hull_points(
        camera_data, loaded_masks, (-args.hull_bound, args.hull_bound),
        args.hull_resolution, args.min_view_ratio,
    )
    rng = np.random.default_rng(0)
    if len(points) > 30_000:
        points = points[rng.choice(len(points), 30_000, replace=False)]
    sparse = args.output / "sparse" / "0"
    sparse.mkdir(parents=True, exist_ok=True)
    # LiteGS creates this cache from points3D.txt. It must not survive a
    # regenerated initialization cloud.
    cached_ply = sparse / "points3D.ply"
    if cached_ply.exists():
        cached_ply.unlink()
    (sparse / "cameras.txt").write_text("\n".join(camera_lines) + "\n")
    (sparse / "images.txt").write_text("\n".join(image_lines) + "\n")
    point_lines = [
        f"{point_id} {xyz[0]:.12g} {xyz[1]:.12g} {xyz[2]:.12g} 127 127 127 0"
        for point_id, xyz in enumerate(points, start=1)
    ]
    (sparse / "points3D.txt").write_text("\n".join(point_lines) + "\n")
    shutil.copy2(args.input / "cameras.npz", args.output / "cameras.npz")
    print(f"Prepared {len(images)} DTU views and {len(points)} shared visual-hull surface points at {args.output}")


if __name__ == "__main__":
    main()
