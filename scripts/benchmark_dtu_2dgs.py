#!/usr/bin/env python3
"""Run and summarize matched vanilla LiteGS/2DGS DTU experiments."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str], cwd: Path = ROOT) -> float:
    started = time.perf_counter()
    subprocess.run(command, cwd=cwd, check=True)
    return time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--scan-input", type=Path, required=True)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scan", type=int, default=24)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--resolution", type=int, default=4)
    parser.add_argument("--skip-training", action="store_true")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    records: dict[str, dict] = {}
    for name, dimension in (("vanilla", 3), ("2dgs", 2)):
        output = args.output_root / name
        if not args.skip_training:
            if output.exists() and any(output.iterdir()):
                raise FileExistsError(f"Refusing to overwrite non-empty benchmark output: {output}")
            entry = ROOT / ("example_train_2dgs.py" if dimension == 2 else "example_train.py")
            command = [
                sys.executable, str(entry), "-s", str(args.dataset), "-m", str(output),
                "--resolution", str(args.resolution), "--iterations", str(args.iterations),
                "--cluster_size", "0", "--densify_from", "999999", "--test_epochs", "999999",
            ]
            wall_seconds = run(command)
        else:
            wall_seconds = float("nan")
        prediction = output / "point_cloud" / "finish" / "point_cloud.ply"
        metric_path = output / "dtu_metrics.json"
        run([
            sys.executable, str(ROOT / "scripts" / "evaluate_dtu_gaussians.py"),
            "--prediction", str(prediction), "--scan-input", str(args.scan_input),
            "--official-root", str(args.official_root), "--scan", str(args.scan),
            "--output", str(metric_path),
        ])
        training_stats_path = output / "training_stats.json"
        records[name] = {
            "wall_seconds": wall_seconds,
            "training": json.loads(training_stats_path.read_text()) if training_stats_path.exists() else None,
            "geometry": json.loads(metric_path.read_text()),
        }
        rgb_output = output / "rgb_evaluation"
        run([
            sys.executable, str(ROOT / "scripts" / "evaluate_rgb.py"),
            "--dataset", str(args.dataset), "--model", str(prediction),
            "--output", str(rgb_output), "--dimension", str(dimension),
            "--resolution", str(args.resolution),
        ])
        records[name]["rgb"] = json.loads((rgb_output / "rgb_metrics.json").read_text())
    vanilla, two_d = records["vanilla"], records["2dgs"]
    records["comparison"] = {
        "geometry_chamfer_improvement_percent": 100.0 * (
            vanilla["geometry"]["chamfer_l1_mm"] - two_d["geometry"]["chamfer_l1_mm"]
        ) / vanilla["geometry"]["chamfer_l1_mm"],
        "training_time_ratio_2dgs_over_vanilla": (
            two_d["training"]["training_seconds"] / vanilla["training"]["training_seconds"]
            if two_d["training"] and vanilla["training"] else None
        ),
        "foreground_psnr_delta_2dgs_minus_vanilla_db": (
            two_d["rgb"]["mean_foreground_psnr"]
            - vanilla["rgb"]["mean_foreground_psnr"]
        ),
    }
    report = args.output_root / "comparison.json"
    report.write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(records, indent=2))
    print(f"Saved benchmark report to {report}")


if __name__ == "__main__":
    main()
