#!/usr/bin/env python3
"""Generate per-frame depth PNGs for a Splatter360 scene using Depth Anything V2."""

import argparse
import sys
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Allow importing the third-party Depth Anything V2 implementation we vendored.
REPO_ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY_DIR = REPO_ROOT / "third_party" / "depth_anything_v2"
if THIRD_PARTY_DIR.exists():
    sys.path.insert(0, str(THIRD_PARTY_DIR))
else:
    raise RuntimeError(
        "Expected Depth Anything V2 repo under third_party/depth_anything_v2."
        " Run `git clone https://github.com/arplaboratory/depth_anything_v2 third_party/depth_anything_v2``"
    )

from depth_anything_v2.dpt import DepthAnythingV2  # noqa: E402


MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene-root",
        type=Path,
        required=True,
        help="Path to a Splatter360 scene directory (must contain pano/ and pano_depth/).",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=REPO_ROOT / "checkpoints" / "depth_anything_v2_vits.pth",
        help="Path to the Depth Anything V2 checkpoint (.pth).",
    )
    parser.add_argument(
        "--encoder",
        choices=MODEL_CONFIGS.keys(),
        default="vits",
        help="Encoder variant that matches the provided weights (default: vits).",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=518,
        help="Short side fed into the network (must be multiple of 14).",
    )
    parser.add_argument(
        "--scale-factor",
        type=float,
        default=1000.0,
        help="Multiply normalized depth by this factor before saving as uint16 (default: 1000 for millimetres).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="torch device string (cuda/cuda:0/mps/cpu). Falls back gracefully if unavailable.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute and overwrite existing depth PNGs.",
    )
    return parser.parse_args()


def resolve_device(device_str: str) -> torch.device:
    requested = torch.device(device_str)
    if requested.type == "cuda" and not torch.cuda.is_available():
        print("[generate_depth_maps] CUDA not available, falling back to CPU.")
        return torch.device("cpu")
    if requested.type == "mps" and not torch.backends.mps.is_available():
        print("[generate_depth_maps] MPS not available, falling back to CPU.")
        return torch.device("cpu")
    return requested


def load_model(weights: Path, encoder: str, device: torch.device) -> DepthAnythingV2:
    if encoder not in MODEL_CONFIGS:
        raise ValueError(f"Unknown encoder '{encoder}'.")

    model = DepthAnythingV2(**MODEL_CONFIGS[encoder])
    state = torch.load(weights, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError(f"Missing keys when loading weights: {missing[:10]}")
    if unexpected:
        print(f"[generate_depth_maps] Warning: unexpected keys ignored: {unexpected[:5]}")

    model.to(device)
    model.eval()
    return model


def list_pano_frames(pano_dir: Path) -> List[Path]:
    return sorted(pano_dir.glob("*.png"))


def main() -> None:
    args = parse_args()
    scene_root = args.scene_root.resolve()
    pano_dir = scene_root / "pano"
    depth_dir = scene_root / "pano_depth"

    if not pano_dir.exists():
        raise FileNotFoundError(f"{pano_dir} not found. Did you run prepare_my360_dataset.py?")
    depth_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    model = load_model(args.weights, args.encoder, device)

    frame_paths = list_pano_frames(pano_dir)
    if not frame_paths:
        raise RuntimeError(f"No panorama frames found under {pano_dir}")

    print(f"[generate_depth_maps] Frames: {len(frame_paths)} | Device: {device}")

    for frame_path in tqdm(frame_paths, desc="Estimating depth"):
        depth_path = depth_dir / frame_path.name
        if depth_path.exists() and not args.overwrite:
            continue

        image_bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"Failed to read image: {frame_path}")

        with torch.no_grad():
            image_tensor, (orig_h, orig_w) = model.image2tensor(image_bgr, input_size=args.input_size)
            image_tensor = image_tensor.to(device)
            _, depth_pred = model(image_tensor)
            depth_resized = F.interpolate(depth_pred, size=(orig_h, orig_w), mode="bilinear", align_corners=True)
            depth = depth_resized[0, 0].cpu().numpy()

        depth = depth - depth.min()
        denom = depth.max() - depth.min()
        if denom > 1e-6:
            depth = depth / denom
        else:
            depth = np.zeros_like(depth)

        depth_mm = np.clip(depth * args.scale_factor, 0, np.iinfo(np.uint16).max).astype(np.uint16)
        if not cv2.imwrite(str(depth_path), depth_mm):
            raise RuntimeError(f"Failed to write depth map: {depth_path}")

    print("[generate_depth_maps] Done.")


if __name__ == "__main__":
    main()
