#!/usr/bin/env python3
"""Convert per-scene meta files into chunked torch datasets."""

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, List

import torch

DATASET_NAME = "my360"
BASEDIR = Path("/srv/store/docker-users/thesis/marten1/data")
INPUT_DATASET = BASEDIR / f"{DATASET_NAME}_dataset"
OUTPUT_DATASET = BASEDIR / f"{DATASET_NAME}_dataset_pt"
STAGES: Iterable[str] = ("train",)
TARGET_BYTES_PER_CHUNK = int(1e8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-pickle",
        action="store_true",
        help="Load meta.pt with weights_only=False (unsafe; only use with trusted data).",
    )
    return parser.parse_args()


def load_meta(path: Path, allow_pickle: bool) -> dict:
    if allow_pickle:
        return torch.load(path, weights_only=False)
    return torch.load(path)


def save_chunk(examples: List[dict], stage_out_dir: Path, chunk_index: int, index_map: dict) -> None:
    if not examples:
        return

    chunk_name = f"{chunk_index:06d}.torch"
    chunk_path = stage_out_dir / chunk_name
    torch.save(examples, chunk_path)

    for example in examples:
        key = example.get("key")
        if key is None:
            raise KeyError("Example missing 'key'.")
        if key in index_map:
            raise ValueError(f"Duplicate key: {key}")
        index_map[key] = chunk_name


def process_stage(stage: str, allow_pickle: bool) -> None:
    stage_input_dir = INPUT_DATASET / stage
    if not stage_input_dir.exists():
        raise FileNotFoundError(f"{stage_input_dir} does not exist")

    stage_out_dir = OUTPUT_DATASET / stage
    stage_out_dir.mkdir(parents=True, exist_ok=True)

    meta_paths = sorted(stage_input_dir.glob("*/meta.pt"))
    if not meta_paths:
        raise RuntimeError(f"No meta.pt files found under {stage_input_dir}")

    examples: List[dict] = []
    chunk_size = 0
    chunk_index = 0
    index_map: dict = {}

    for meta_path in meta_paths:
        example = load_meta(meta_path, allow_pickle)
        examples.append(example)
        chunk_size += os.path.getsize(meta_path)

        if chunk_size >= TARGET_BYTES_PER_CHUNK:
            save_chunk(examples, stage_out_dir, chunk_index, index_map)
            chunk_index += 1
            examples = []
            chunk_size = 0

    if examples:
        save_chunk(examples, stage_out_dir, chunk_index, index_map)

    index_path = stage_out_dir / "index.json"
    index_path.write_text(json.dumps(index_map, indent=2))
    print(f"[convert] Stage '{stage}' → {len(index_map)} scenes, output {stage_out_dir}")


def main() -> None:
    args = parse_args()
    for stage in STAGES:
        process_stage(stage, args.allow_pickle)


if __name__ == "__main__":
    main()
