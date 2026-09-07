#!/usr/bin/env python3
"""Create an aspect-correct four-sensor MP4 from a retained UniVTAC HDF5 episode.

This is an artifact-only operation: it does not import or launch Isaac Sim and
does not change policy observations, physics, or task data.
"""

import argparse
import json
from pathlib import Path

import cv2
import h5py
import numpy as np


CAMERA_PATHS = {
    "Head RGB": "observation/head/rgb",
    "Wrist RGB": "observation/wrist/rgb",
    "Left tactile": "tactile/left_tactile/rgb_marker",
    "Right tactile": "tactile/right_tactile/rgb_marker",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_hdf5", type=Path)
    parser.add_argument("output_mp4", type=Path)
    parser.add_argument("--fps", type=float, default=30.0, help="Playback rate; this is not wall-clock simulator FPS.")
    parser.add_argument("--task-label", default="UniVTAC lift_can - Isaac Sim 6 migration")
    return parser.parse_args()


def decode_frame(value: bytes | np.bytes_) -> np.ndarray:
    encoded = np.frombuffer(bytes(value), dtype=np.uint8)
    frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("failed to decode an HDF5 image frame")
    return frame


def fit_to_cell(frame: np.ndarray, cell_width: int = 480, cell_height: int = 270) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = min(cell_width / width, cell_height / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(frame, (resized_width, resized_height), interpolation=interpolation)
    cell = np.zeros((cell_height, cell_width, 3), dtype=np.uint8)
    top = (cell_height - resized_height) // 2
    left = (cell_width - resized_width) // 2
    cell[top : top + resized_height, left : left + resized_width] = resized
    return cell


def label_cell(cell: np.ndarray, label: str) -> np.ndarray:
    output = cell.copy()
    cv2.rectangle(output, (0, 0), (210, 32), (18, 18, 18), thickness=-1)
    cv2.putText(output, label, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    return output


def decode_text(value: bytes | np.bytes_ | str) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8", errors="replace")
    return str(value)


def main() -> None:
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if not args.input_hdf5.is_file():
        raise FileNotFoundError(args.input_hdf5)
    if args.output_mp4.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output_mp4}")

    args.output_mp4.parent.mkdir(parents=True, exist_ok=True)
    frame_width, frame_height = 960, 540
    writer = cv2.VideoWriter(
        str(args.output_mp4),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (frame_width, frame_height),
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not open its MP4 writer")

    manifest = {
        "input_hdf5": str(args.input_hdf5),
        "output_mp4": str(args.output_mp4),
        "fps": args.fps,
        "playback_fps_note": "Playback FPS is presentation timing, not measured simulator wall-clock throughput.",
        "layout": [["Head RGB", "Wrist RGB"], ["Left tactile", "Right tactile"]],
        "frame_size": [frame_width, frame_height],
        "source_datasets": CAMERA_PATHS,
    }

    try:
        with h5py.File(args.input_hdf5, "r") as h5_file:
            missing = [path for path in CAMERA_PATHS.values() if path not in h5_file]
            if missing:
                raise KeyError(f"missing required HDF5 datasets: {missing}")
            counts = {label: len(h5_file[path]) for label, path in CAMERA_PATHS.items()}
            if len(set(counts.values())) != 1:
                raise ValueError(f"sensor frame counts differ: {counts}")
            frame_count = next(iter(counts.values()))
            steps = h5_file["step"] if "step" in h5_file else None
            atom_ids = h5_file["atom/id"] if "atom/id" in h5_file else None
            atom_tags = h5_file["atom/tag"] if "atom/tag" in h5_file else None

            for index in range(frame_count):
                cells = [
                    label_cell(fit_to_cell(decode_frame(h5_file[path][index])), label)
                    for label, path in CAMERA_PATHS.items()
                ]
                montage = np.vstack((np.hstack(cells[:2]), np.hstack(cells[2:])))
                step = int(steps[index]) if steps is not None else index
                atom_id = int(atom_ids[index]) if atom_ids is not None else -1
                atom_tag = decode_text(atom_tags[index]) if atom_tags is not None else "unknown"
                footer = (
                    f"{args.task_label} | saved frame {index + 1}/{frame_count} | "
                    f"physics step {step} | atom {atom_id}: {atom_tag}"
                )
                cv2.rectangle(montage, (0, frame_height - 29), (frame_width, frame_height), (18, 18, 18), thickness=-1)
                cv2.putText(
                    montage,
                    footer,
                    (10, frame_height - 9),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.46,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                writer.write(montage)
    except Exception:
        writer.release()
        args.output_mp4.unlink(missing_ok=True)
        raise
    writer.release()

    manifest["frame_count"] = frame_count
    manifest["duration_seconds"] = frame_count / args.fps
    manifest["source_frame_counts"] = counts
    manifest_path = args.output_mp4.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
