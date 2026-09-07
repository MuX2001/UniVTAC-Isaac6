#!/usr/bin/env python3
"""Compare policy-camera image statistics in two UniVTAC HDF5 episodes.

The comparison is distributional rather than pixel-aligned: independently
generated simulator episodes can have different frame counts and trajectories.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import h5py
import numpy as np


CAMERA_PATHS = ("observation/head/rgb", "observation/wrist/rgb")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def decode_rgb(value: bytes | np.bytes_) -> np.ndarray:
    encoded = np.frombuffer(bytes(value), dtype=np.uint8)
    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("failed to decode HDF5 RGB frame")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def camera_metrics(dataset: h5py.Dataset) -> dict:
    pixel_sum = np.zeros(3, dtype=np.float64)
    pixel_square_sum = np.zeros(3, dtype=np.float64)
    pixel_count = 0
    luma_values = []
    sharpness_values = []
    temporal_mae_values = []
    dark_count = 0
    bright_count = 0
    shapes = set()
    previous = None

    for value in dataset:
        frame = decode_rgb(value)
        shapes.add(tuple(int(item) for item in frame.shape))
        flat = frame.reshape(-1, 3).astype(np.float64)
        pixel_sum += flat.sum(axis=0)
        pixel_square_sum += np.square(flat).sum(axis=0)
        pixel_count += flat.shape[0]
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        luma_values.append(float(gray.mean()))
        sharpness_values.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
        dark_count += int((gray <= 5).sum())
        bright_count += int((gray >= 250).sum())
        if previous is not None:
            if previous.shape != frame.shape:
                raise ValueError(f"shape changed inside one camera stream: {previous.shape} -> {frame.shape}")
            temporal_mae_values.append(float(np.abs(frame.astype(np.int16) - previous.astype(np.int16)).mean()))
        previous = frame

    channel_mean = pixel_sum / pixel_count
    channel_variance = np.maximum(pixel_square_sum / pixel_count - np.square(channel_mean), 0.0)
    grayscale_pixel_count = pixel_count
    return {
        "frames": len(dataset),
        "decoded_shapes_hwc": [list(shape) for shape in sorted(shapes)],
        "rgb_mean": channel_mean.tolist(),
        "rgb_std": np.sqrt(channel_variance).tolist(),
        "frame_luma_mean": float(np.mean(luma_values)),
        "frame_luma_std": float(np.std(luma_values)),
        "mean_laplacian_variance": float(np.mean(sharpness_values)),
        "mean_consecutive_frame_mae": float(np.mean(temporal_mae_values)) if temporal_mae_values else None,
        "near_black_fraction": dark_count / grayscale_pixel_count,
        "near_white_fraction": bright_count / grayscale_pixel_count,
    }


def episode_metrics(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as h5_file:
        missing = [camera_path for camera_path in CAMERA_PATHS if camera_path not in h5_file]
        if missing:
            raise KeyError(f"{path} is missing camera datasets: {missing}")
        return {camera_path: camera_metrics(h5_file[camera_path]) for camera_path in CAMERA_PATHS}


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output}")
    result = {
        "interpretation": (
            "Distributional policy-input comparison; trajectories are not frame-aligned, so differences do not by "
            "themselves identify a camera-geometry fault."
        ),
        "reference": {"path": str(args.reference), "cameras": episode_metrics(args.reference)},
        "candidate": {"path": str(args.candidate), "cameras": episode_metrics(args.candidate)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
