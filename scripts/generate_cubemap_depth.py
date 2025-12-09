#!/usr/bin/env python3
"""Convert pano depth PNGs into cubemap depth tensors required by Splatter360."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.geometry.util import Equirec2Cube  # noqa: E402


def reorder_faces(cube_array: np.ndarray) -> np.ndarray:
    """Match the face ordering expected by dataset_hm3d.py."""
    reordered = np.empty_like(cube_array)
    reordered[0] = cube_array[4]
    reordered[1] = cube_array[2]
    reordered[2] = cube_array[3]
    reordered[3] = cube_array[0]
    reordered[4] = cube_array[1]
    reordered[5] = cube_array[5]
    reordered[0] = np.flip(reordered[0], axis=(0, 1))
    reordered[5] = np.flip(reordered[5], axis=(0, 1))
    return reordered


def process_scene(scene_dir: Path, depth_scale: float, overwrite: bool) -> None:
    meta_path = scene_dir / "meta.pt"
    if not meta_path.exists():
        raise FileNotFoundError(f"{meta_path} not found; run convert_cubemaps_mp.py first.")

    meta = torch.load(meta_path, weights_only=False)
    cube_h, cube_w = [int(x) for x in meta["cube_shape"].tolist()]

    pano_dir = scene_dir / "pano"
    depth_dir = scene_dir / "pano_depth"
    cubemap_dir = scene_dir / "cubemaps_depth"
    cubemap_dir.mkdir(parents=True, exist_ok=True)

    pano_paths = sorted(pano_dir.glob("*.png"))
    if not pano_paths:
        raise RuntimeError(f"No panorama frames found in {pano_dir}")

    sample_rgb = cv2.imread(str(pano_paths[0]), cv2.IMREAD_COLOR)
    if sample_rgb is None:
        raise RuntimeError(f"Failed to read {pano_paths[0]}")
    equ_h, equ_w = sample_rgb.shape[:2]

    e2c = Equirec2Cube(equ_h, equ_w, cube_h)

    for pano_path in pano_paths:
        depth_path = depth_dir / pano_path.name
        if not depth_path.exists():
            raise FileNotFoundError(f"Missing depth map: {depth_path}")

        cube_path = cubemap_dir / f"{pano_path.stem}.torch"
        if cube_path.exists() and not overwrite:
            continue

        rgb = cv2.imread(str(pano_path), cv2.IMREAD_COLOR)
        depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth_raw is None:
            raise RuntimeError(f"Failed to read depth map {depth_path}")

        depth_m = depth_raw.astype(np.float32) / depth_scale
        depth_m = depth_m[..., None]

        _, cube_depth = e2c.run(rgb, depth_m)
        cube_depth = cube_depth[..., 0]
        cube_depth = cube_depth.reshape(cube_h, 6, cube_w)
        cube_depth = np.transpose(cube_depth, (1, 0, 2))  # (6, h, w)
        cube_depth = cube_depth[..., None]
        cube_depth = reorder_faces(cube_depth)

        torch.save(torch.from_numpy(cube_depth.astype(np.float32)), cube_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True,
                        help="Path to the my360_dataset directory (containing train/, test/ etc.)")
    parser.add_argument("--depth-scale", type=float, default=1000.0,
                        help="Divide raw depth values by this factor to convert to meters (default: 1000).")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing cubemap depth tensors.")
    args = parser.parse_args()

    for stage_dir in sorted(args.dataset_root.iterdir()):
        if not stage_dir.is_dir():
            continue
        for scene_dir in sorted(stage_dir.iterdir()):
            if not scene_dir.is_dir():
                continue
            print(f"[generate_cubemap_depth] {stage_dir.name}/{scene_dir.name}")
            process_scene(scene_dir, args.depth_scale, args.overwrite)


if __name__ == "__main__":
    main()
