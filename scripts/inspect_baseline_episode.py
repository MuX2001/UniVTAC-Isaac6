#!/usr/bin/env python3
"""Validate a raw Isaac Sim 4 UniVTAC baseline episode and write a manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


REQUIRED_DATASETS = (
    "observation/head/rgb",
    "observation/head/depth",
    "observation/wrist/rgb",
    "observation/wrist/depth",
    "tactile/left_tactile/rgb",
    "tactile/left_tactile/depth",
    "tactile/right_tactile/rgb",
    "tactile/right_tactile/depth",
    "atom/id",
    "atom/tag",
    "step",
)


def dataset_summary(dataset: h5py.Dataset) -> dict[str, object]:
    data = dataset[()]
    summary: dict[str, object] = {"shape": list(dataset.shape), "dtype": str(dataset.dtype)}
    if np.issubdtype(data.dtype, np.number):
        summary["min"] = float(np.nanmin(data))
        summary["max"] = float(np.nanmax(data))
    return summary


def tactile_depth_summary(dataset: h5py.Dataset) -> dict[str, float | int]:
    """Summarize per-pad deformation without changing the stored raw frames."""
    depth = dataset[()].astype(np.float64, copy=False)
    flattened = depth.reshape(depth.shape[0], -1)
    spatial_range = np.ptp(flattened, axis=1)
    temporal_delta = np.mean(np.abs(np.diff(depth, axis=0)), axis=tuple(range(1, depth.ndim)))
    return {
        "frames_with_spatial_variation": int(np.count_nonzero(spatial_range > 0)),
        "spatial_range_min": float(np.min(spatial_range)),
        "spatial_range_max": float(np.max(spatial_range)),
        "temporal_delta_mean": float(np.mean(temporal_delta)) if len(temporal_delta) else 0.0,
        "temporal_delta_max": float(np.max(temporal_delta)) if len(temporal_delta) else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path, help="Collected .hdf5 episode")
    parser.add_argument("--output", type=Path, help="Manifest path (default: <episode>.manifest.json)")
    args = parser.parse_args()

    if not args.episode.is_file():
        raise SystemExit(f"Episode does not exist: {args.episode}")
    output = args.output or args.episode.with_suffix(".manifest.json")

    with h5py.File(args.episode, "r") as h5:
        missing = [path for path in REQUIRED_DATASETS if path not in h5]
        if missing:
            raise SystemExit(f"Baseline is missing required datasets: {', '.join(missing)}")

        rgb_paths = [path for path in REQUIRED_DATASETS if path.endswith("/rgb")]
        encoded = [path for path in rgb_paths if h5[path].dtype.kind in {"S", "O", "U"}]
        if encoded:
            raise SystemExit(
                "RGB is encoded instead of raw; run with encode_images: false. "
                f"Encoded datasets: {', '.join(encoded)}"
            )

        frames = int(h5["step"].shape[0])
        if frames == 0:
            raise SystemExit("Episode has no saved frames")

        stages: list[dict[str, object]] = []
        atom_ids = h5["atom/id"][()]
        atom_tags = h5["atom/tag"][()]
        steps = h5["step"][()]
        for frame, (atom_id, atom_tag, step) in enumerate(zip(atom_ids, atom_tags, steps, strict=True)):
            tag = atom_tag.decode("utf-8") if isinstance(atom_tag, bytes) else str(atom_tag)
            if frame == 0 or atom_id != atom_ids[frame - 1] or tag != stages[-1]["tag"]:
                stages.append({"first_frame": frame, "atom_id": int(atom_id), "tag": tag, "step": int(step)})

        manifest = {
            "episode": str(args.episode.resolve()),
            "frames": frames,
            "raw_rgb": True,
            "datasets": {path: dataset_summary(h5[path]) for path in REQUIRED_DATASETS},
            "tactile_depth": {
                "left_tactile": tactile_depth_summary(h5["tactile/left_tactile/depth"]),
                "right_tactile": tactile_depth_summary(h5["tactile/right_tactile/depth"]),
            },
            "stages": stages,
        }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
