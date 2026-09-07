#!/usr/bin/env python3
"""Extract compact visual/tactile evidence from one public UniVTAC HDF5 episode.

This is a read-only helper for comparing a known-good published trajectory with
a local runtime validation.  It does not create an Isaac Sim application or
modify the episode.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


def _to_jsonable(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _array_stats(value: np.ndarray) -> dict[str, float | int | list[int] | str | None]:
    value = np.asarray(value)
    finite = np.isfinite(value)
    finite_values = value[finite]
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "finite_fraction": float(finite.mean()) if value.size else 1.0,
        "nan_count": int(np.isnan(value).sum()),
        "inf_count": int(np.isinf(value).sum()),
        "min": float(finite_values.min()) if finite_values.size else None,
        "max": float(finite_values.max()) if finite_values.size else None,
        "mean": float(finite_values.mean()) if finite_values.size else None,
        "std": float(finite_values.std()) if finite_values.size else None,
    }


def _decode_rgb(value) -> np.ndarray:
    if isinstance(value, (bytes, bytearray, np.bytes_)):
        encoded = bytes(value)
    else:
        value = np.asarray(value)
        if value.ndim == 3 and value.shape[-1] in (3, 4):
            return value[..., :3].astype(np.uint8, copy=False)
        encoded = value.astype(np.uint8, copy=False).tobytes()
    with Image.open(io.BytesIO(encoded)) as image:
        return np.asarray(image.convert("RGB"))


def _find_tactile_dataset(h5_file: h5py.File, side: str, leaf: str) -> h5py.Dataset:
    for sensor_name in (f"{side}_gsmini", f"{side}_tactile"):
        path = f"tactile/{sensor_name}/{leaf}"
        if path in h5_file:
            return h5_file[path]
    raise KeyError(f"No {side} tactile {leaf!r} dataset found")


def _center_has_contact(depth: np.ndarray) -> bool:
    depth = np.asarray(depth)
    center = depth[50:-50, 50:-50] if min(depth.shape[-2:]) > 100 else depth
    finite = center[np.isfinite(center)]
    return bool(finite.size and float(finite.min()) != float(finite.max()))


def _capture_state(group: h5py.Group, index: int, episode_length: int) -> dict:
    captured = {}
    for name, node in group.items():
        if isinstance(node, h5py.Group):
            captured[name] = _capture_state(node, index, episode_length)
        elif isinstance(node, h5py.Dataset):
            value = node[index] if node.ndim and node.shape[0] == episode_length else node[()]
            captured[name] = _to_jsonable(value)
    return captured


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Downloaded UniVTAC HDF5 episode")
    parser.add_argument("--output-dir", type=Path, required=True, help="Destination for extracted reference evidence")
    parser.add_argument("--source-url", default="", help="Official URL recorded in the manifest")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.input, "r") as h5_file:
        episode_length = int(h5_file["embodiment/joint"].shape[0])
        left_depth = _find_tactile_dataset(h5_file, "left", "depth")
        right_depth = _find_tactile_dataset(h5_file, "right", "depth")
        contact_index = next(
            (
                index
                for index in range(episode_length)
                if _center_has_contact(left_depth[index]) and _center_has_contact(right_depth[index])
            ),
            None,
        )
        frames = {"first": 0, "contact": contact_index, "last": episode_length - 1}
        frames = {label: index for label, index in frames.items() if index is not None}

        manifest = {
            "input": str(args.input),
            "source_url": args.source_url,
            "episode_length": episode_length,
            "selected_frames": frames,
            "fields": {},
        }
        for label, index in frames.items():
            frame_dir = args.output_dir / label
            frame_dir.mkdir(exist_ok=True)
            state = {
                "frame_index": index,
                "embodiment": _capture_state(h5_file["embodiment"], index, episode_length),
                "actor": _capture_state(h5_file["actor"], index, episode_length),
                "atom": _capture_state(h5_file["atom"], index, episode_length),
                "tactile_center_contact": {},
            }
            for camera_name in ("head", "wrist"):
                dataset = h5_file[f"observation/{camera_name}/rgb"]
                rgb = _decode_rgb(dataset[index])
                Image.fromarray(rgb).save(frame_dir / f"{camera_name}_rgb.png")
                state[f"{camera_name}_rgb"] = _array_stats(rgb)

            for side in ("left", "right"):
                depth = _find_tactile_dataset(h5_file, side, "depth")[index]
                np.save(frame_dir / f"{side}_tactile_depth.npy", depth)
                state["tactile_center_contact"][side] = _center_has_contact(depth)
                state[f"{side}_tactile_depth"] = _array_stats(depth)
                for leaf in ("rgb", "rgb_marker"):
                    rgb = _decode_rgb(_find_tactile_dataset(h5_file, side, leaf)[index])
                    Image.fromarray(rgb).save(frame_dir / f"{side}_tactile_{leaf}.png")
                    state[f"{side}_tactile_{leaf}"] = _array_stats(rgb)
            with open(frame_dir / "state.json", "w", encoding="utf-8") as file:
                json.dump(state, file, indent=2, allow_nan=False)

        manifest["fields"] = {
            "embodiment/joint": list(h5_file["embodiment/joint"].shape),
            "observation/head/rgb": list(h5_file["observation/head/rgb"].shape),
            "observation/wrist/rgb": list(h5_file["observation/wrist/rgb"].shape),
            "left_tactile/depth": list(left_depth.shape),
            "right_tactile/depth": list(right_depth.shape),
        }
    with open(args.output_dir / "manifest.json", "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
