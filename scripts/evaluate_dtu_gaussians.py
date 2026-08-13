#!/usr/bin/env python3
"""Evaluate Gaussian primitive centers with the DTU crop-aware point metric."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from plyfile import PlyData
from scipy.io import loadmat
from sklearn.neighbors import NearestNeighbors


def load_xyz(path: Path) -> np.ndarray:
    vertex = PlyData.read(path).elements[0]
    return np.stack([vertex[name] for name in ("x", "y", "z")], axis=1).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--scan-input", type=Path, required=True)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--scan", type=int, default=24)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-dist-mm", type=float, default=20.0)
    args = parser.parse_args()

    predicted_normalized = load_xyz(args.prediction)
    scale = np.load(args.scan_input / "cameras.npz")["scale_mat_0"].astype(np.float32)
    predicted_h = np.concatenate([predicted_normalized, np.ones((len(predicted_normalized), 1), np.float32)], axis=1)
    predicted = (predicted_h @ scale.T)[:, :3]
    # Deterministic 0.2 mm voxel downsampling used by the official protocol.
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(predicted))
    predicted = np.asarray(cloud.voxel_down_sample(0.2).points)

    obs_data = loadmat(args.official_root / "ObsMask" / f"ObsMask{args.scan}_10.mat")
    obs, bounds, resolution = obs_data["ObsMask"], obs_data["BB"].astype(np.float32), obs_data["Res"]
    inbound = ((predicted >= bounds[:1] - 60) & (predicted < bounds[1:] + 120)).all(axis=1)
    candidate = predicted[inbound]
    grid = np.around((candidate - bounds[:1]) / resolution).astype(np.int32)
    grid_ok = ((grid >= 0) & (grid < np.asarray(obs.shape)[None])).all(axis=1)
    candidate = candidate[grid_ok]
    grid = grid[grid_ok]
    predicted_observed = candidate[obs[grid[:, 0], grid[:, 1], grid[:, 2]].astype(bool)]

    gt_path = args.official_root / "Points" / "stl" / f"stl{args.scan:03d}_total.ply"
    gt = np.asarray(o3d.io.read_point_cloud(str(gt_path)).points)
    accuracy = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(gt).kneighbors(predicted_observed)[0][:, 0]
    plane = loadmat(args.official_root / "ObsMask" / f"Plane{args.scan}.mat")["P"].reshape(4)
    gt_above = gt[(np.c_[gt, np.ones(len(gt))] * plane).sum(axis=1) > 0]
    completeness = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(candidate).kneighbors(gt_above)[0][:, 0]
    accuracy = accuracy[accuracy < args.max_dist_mm]
    completeness = completeness[completeness < args.max_dist_mm]
    result = {
        "geometry_representation": "gaussian_primitive_centers",
        "protocol": "DTU observation-mask/ground-plane crop, 0.2 mm prediction downsample, 20 mm outlier cap",
        "accuracy_mm": float(accuracy.mean()),
        "completeness_mm": float(completeness.mean()),
        "chamfer_l1_mm": float((accuracy.mean() + completeness.mean()) / 2),
        "predicted_points": int(len(predicted)),
        "observed_predicted_points": int(len(predicted_observed)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
