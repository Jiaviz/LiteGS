#!/usr/bin/env python3
"""Render a LiteGS PLY and report full/foreground PSNR per view."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import litegs
from litegs.render import surfel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--images", type=Path, default=Path("images"))
    parser.add_argument("--dimension", type=int, choices=(2, 3), required=True)
    parser.add_argument("--resolution", type=int, default=4)
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument("--scale-multiplier", type=float, default=1.0)
    parser.add_argument("--max-images", type=int, default=0)
    args = parser.parse_args()

    cameras, frames, _, _ = litegs.io_manager.load_colmap_result(str(args.dataset), str(args.images))
    for frame in frames:
        frame.load_image(args.resolution)
    dataset = litegs.data.CameraFrameDataset(
        cameras, frames, args.resolution, True, mask_root=str(args.dataset / "masks")
    )
    tensors = [torch.tensor(value, dtype=torch.float32, device="cuda") for value in
               litegs.io_manager.load_ply(str(args.model), args.sh_degree)]
    xyz, scale, rot, sh0, shrest, opacity = tensors
    scale = scale.clone()
    scale[:args.dimension] += float(np.log(args.scale_multiplier))
    pp = litegs.arguments.PipelineParams.get_class_default_obj()
    pp.gaussian_dimension = args.dimension
    pp.cluster_size = 0
    pp.enable_transmitance = True
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    limit = len(dataset) if args.max_images <= 0 else min(args.max_images, len(dataset))
    with torch.inference_mode():
        for index in range(limit):
            view, proj, frustum, gt, mask, _, _ = dataset[index]
            def cuda_tensor(value):
                return value.cuda() if torch.is_tensor(value) else torch.as_tensor(value, device="cuda")
            view, proj, frustum = (
                cuda_tensor(view)[None], cuda_tensor(proj)[None], cuda_tensor(frustum)[None]
            )
            gt = gt[None].float() / 255.0
            mask = mask[None].float()
            _, cx, cs, cr, c0, crest, co = litegs.render.render_preprocess(
                None, None, frustum, xyz, scale, rot, sh0, shrest, opacity, None, pp
            )
            if args.dimension == 2:
                rendered = surfel.render(
                    view, proj, cx, cs, cr, c0, crest, co, args.sh_degree,
                    gt.shape[2:], pp, compute_surface_normal=False,
                )[0]
            else:
                rendered = litegs.render.render(
                    view, proj, cx, cs, cr, c0, crest, co, args.sh_degree, gt.shape[2:], pp
                )[0]
            mse = (rendered - gt).square().mean()
            fg_mse = ((rendered - gt).square() * mask).sum() / (mask.sum() * 3).clamp_min(1)
            record = {
                "image": frames[index].name,
                "psnr": float(-10 * torch.log10(mse.clamp_min(1e-10))),
                "foreground_psnr": float(-10 * torch.log10(fg_mse.clamp_min(1e-10))),
            }
            records.append(record)
            render_u8 = rendered[0].clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy()
            gt_u8 = gt[0].clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy()
            Image.fromarray(np.concatenate((gt_u8, render_u8), axis=1)).save(
                args.output / f"{index:03d}_gt_render.png"
            )
            print(f"[{index + 1}/{limit}] {frames[index].name}: foreground PSNR {record['foreground_psnr']:.3f}")
    summary = {
        "dimension": args.dimension,
        "images": records,
        "mean_psnr": float(np.mean([item["psnr"] for item in records])),
        "mean_foreground_psnr": float(np.mean([item["foreground_psnr"] for item in records])),
    }
    (args.output / "rgb_metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "images"}, indent=2))


if __name__ == "__main__":
    main()
