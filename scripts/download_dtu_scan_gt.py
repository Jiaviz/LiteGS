#!/usr/bin/env python3
"""Range-download only the official DTU files needed for one scan."""

from __future__ import annotations

import argparse
from pathlib import Path

from remotezip import RemoteZip


POINTS_URL = "https://roboimagedata2.compute.dtu.dk/data/MVS/Points.zip"
SAMPLE_URL = "https://roboimagedata2.compute.dtu.dk/data/MVS/SampleSet.zip"


def extract_member(url: str, member: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with RemoteZip(url) as archive, archive.open(member) as source, output.open("wb") as target:
        while chunk := source.read(8 * 1024 * 1024):
            target.write(chunk)
    print(f"Downloaded {member} -> {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan", type=int, default=24)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    extract_member(POINTS_URL, f"Points/stl/stl{args.scan:03d}_total.ply", args.output / "Points/stl" / f"stl{args.scan:03d}_total.ply")
    extract_member(SAMPLE_URL, f"SampleSet/MVS Data/ObsMask/ObsMask{args.scan}_10.mat", args.output / "ObsMask" / f"ObsMask{args.scan}_10.mat")
    extract_member(SAMPLE_URL, f"SampleSet/MVS Data/ObsMask/Plane{args.scan}.mat", args.output / "ObsMask" / f"Plane{args.scan}.mat")


if __name__ == "__main__":
    main()
