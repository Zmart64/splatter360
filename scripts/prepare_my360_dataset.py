#!/usr/bin/env python3
"""Prepare Splatter360-compatible metadata from Potree export files."""

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        required=True,
        help="Directory containing the original panorama JPG files (e.g. pic_*.jpg).",
    )
    parser.add_argument(
        "--meta-dir",
        type=Path,
        required=True,
        help="Directory containing images360_coordinates_timeShifted.txt and related files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help=(
            "Destination root where the script will create the Splatter360 layout,"
            " e.g. /path/to/my360_dataset/train/scene_000"
        ),
    )
    parser.add_argument(
        "--stage",
        default="train",
        help="Dataset split name (train/test). Used only for info in the manifest.",
    )
    parser.add_argument(
        "--scene-name",
        default="scene_000",
        help="Name for the scene folder that will be created under output-root.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Index offset for renaming frames (default: 0).",
    )
    return parser.parse_args()


def read_images_metadata(meta_dir: Path) -> Dict[str, Dict[str, float]]:
    """Read the Potree export table with timestamps and orientation."""
    table_candidates = [
        meta_dir / "images360_coordinates_timeShifted.txt",
        meta_dir / "images360_coordinates.txt",
    ]
    table_path = None
    for candidate in table_candidates:
        if candidate.exists():
            table_path = candidate
            break
    if table_path is None:
        raise FileNotFoundError(
            "Could not find images360_coordinates_timeShifted.txt or fallback file in"
            f" {meta_dir}"
        )

    metadata: Dict[str, Dict[str, float]] = {}
    with table_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            filename = row["File"].strip().strip('"')
            metadata[filename] = {
                "time": float(row.get("Time", "0") or 0.0),
                "x": float(row.get("Long", "0") or 0.0),
                "y": float(row.get("Lat", "0") or 0.0),
                "z": float(row.get("Alt", "0") or 0.0),
                "yaw": float(row.get("Course", "0") or 0.0),
                "pitch": float(row.get("Pitch", "0") or row.get("pitch", "0") or 0.0),
                "roll": float(row.get("Roll", "0") or row.get("roll", "0") or 0.0),
            }
    if not metadata:
        raise RuntimeError(f"No metadata rows parsed from {table_path}")
    return metadata


def rotation_matrix_from_ypr(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """Return a 3x3 camera-to-world rotation matrix from yaw/pitch/roll (Z-Y-X)."""
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    roll = math.radians(roll_deg)

    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)

    # ZYX intrinsic (yaw around Z, pitch around Y, roll around X)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    return (Rz @ Ry @ Rx).astype(np.float32)


def ensure_output_dirs(output_scene_root: Path) -> Tuple[Path, Path, Path]:
    pano_dir = output_scene_root / "pano"
    pano_depth_dir = output_scene_root / "pano_depth"
    cubemap_depth_dir = output_scene_root / "cubemaps_depth"
    pano_dir.mkdir(parents=True, exist_ok=True)
    pano_depth_dir.mkdir(parents=True, exist_ok=True)
    cubemap_depth_dir.mkdir(parents=True, exist_ok=True)
    return pano_dir, pano_depth_dir, cubemap_depth_dir


def convert_images(
    image_files: List[Path],
    pano_dir: Path,
    start_index: int,
) -> List[Path]:
    saved_paths: List[Path] = []
    for idx, src in enumerate(tqdm(image_files, desc="Copying panoramas"), start=start_index):
        dst = pano_dir / f"{idx:06d}.png"
        with Image.open(src) as im:
            im.convert("RGB").save(dst)
        saved_paths.append(dst)
    return saved_paths


def main() -> None:
    args = parse_args()

    raw_files = sorted(args.raw_dir.glob("*.jpg"))
    if not raw_files:
        raise RuntimeError(f"No JPGs found under {args.raw_dir}")

    print(f"[prepare_my360_dataset] Found {len(raw_files)} panorama frames under {args.raw_dir}")

    metadata = read_images_metadata(args.meta_dir)
    print(f"[prepare_my360_dataset] Loaded {len(metadata)} metadata rows")

    # Align metadata to available frames.
    paired: List[Tuple[Path, Dict[str, float]]] = []
    missing = []
    for src in raw_files:
        entry = metadata.get(src.name)
        if entry is None:
            missing.append(src.name)
            continue
        paired.append((src, entry))
    if missing:
        print(
            f"[prepare_my360_dataset] WARNING: {len(missing)} frames missing metadata and will be skipped",
            file=sys.stderr,
        )
    if not paired:
        raise RuntimeError("No frames remain after matching metadata; aborting")

    translations = []
    rotations = []
    timestamps = []

    base_translation = None
    for src, entry in tqdm(paired, desc="Building pose arrays"):
        t = np.array([entry["x"], entry["y"], entry["z"]], dtype=np.float32)
        if base_translation is None:
            base_translation = t.copy()
        t_world = t - base_translation
        translations.append(t_world)
        R = rotation_matrix_from_ypr(entry["yaw"], entry["pitch"], entry["roll"])
        rotations.append(R)
        timestamps.append(entry["time"])

    output_scene_root = args.output_root / args.scene_name
    pano_dir, _, _ = ensure_output_dirs(output_scene_root)
    print(f"[prepare_my360_dataset] Writing panoramas to {pano_dir}")

    convert_images([src for src, _ in paired], pano_dir, args.start_index)

    np.save(output_scene_root / "rotation.npy", np.stack(rotations, axis=0))
    np.save(output_scene_root / "translation.npy", np.stack(translations, axis=0))
    np.save(output_scene_root / "timestamps.npy", np.array(timestamps, dtype=np.float64))

    manifest = {
        "stage": args.stage,
        "scene": args.scene_name,
        "num_frames": len(rotations),
        "base_translation": base_translation.tolist() if base_translation is not None else None,
    }
    (output_scene_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(
        f"[prepare_my360_dataset] Done. Frames: {len(paired)} | Output: {output_scene_root}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[prepare_my360_dataset] Error: {exc}", file=sys.stderr)
        sys.exit(1)
