#!/usr/bin/env python3
"""Summarize an opt-in ``collect_data.py --validation-dir`` tactile probe.

This does not import Isaac Sim or advance a simulation.  Run it inside the
pinned runtime image after a probe has completed:

    /isaac-sim/python.sh scripts/analyze_tactile_validation.py <validation-dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load(path: Path) -> np.ndarray:
    return np.load(path, allow_pickle=False)


def image_metrics(before: np.ndarray, after: np.ndarray) -> dict[str, float | None]:
    before = before.astype(np.float64)
    after = after.astype(np.float64)
    shared_finite = np.isfinite(before) & np.isfinite(after)
    before_flat = before[shared_finite]
    after_flat = after[shared_finite]
    finite_mask_changed = np.isfinite(before) != np.isfinite(after)
    if not before_flat.size:
        return {
            "mean_absolute_difference": None,
            "root_mean_squared_difference": None,
            "correlation": None,
            "shared_finite_fraction": float(shared_finite.mean()),
            "finite_mask_changed_fraction": float(finite_mask_changed.mean()),
        }
    delta = after_flat - before_flat
    if before_flat.std() == 0 or after_flat.std() == 0:
        correlation = None
    else:
        correlation = float(np.corrcoef(before_flat, after_flat)[0, 1])
    return {
        "mean_absolute_difference": float(np.mean(np.abs(delta))),
        "root_mean_squared_difference": float(np.sqrt(np.mean(delta**2))),
        "correlation": correlation,
        "shared_finite_fraction": float(shared_finite.mean()),
        "finite_mask_changed_fraction": float(finite_mask_changed.mean()),
    }


def center_depth_stats(depth: np.ndarray) -> dict[str, float]:
    depth = np.squeeze(depth)
    center = depth[50:-50, 50:-50]
    finite_depth = depth[np.isfinite(depth)]
    finite_center = center[np.isfinite(center)]
    return {
        "min_mm": float(finite_depth.min()),
        "max_mm": float(finite_depth.max()),
        "center_min_mm": float(finite_center.min()),
        "center_max_mm": float(finite_center.max()),
        "center_std_mm": float(finite_center.std()),
    }


def rotation_angle_deg(rotation: np.ndarray) -> float:
    cosine = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def rotation_matrix_from_quat_xyzw(quat: np.ndarray) -> np.ndarray:
    """Return the column-vector rotation matrix for an ``(x, y, z, w)`` quaternion."""
    x, y, z, w = np.asarray(quat, dtype=np.float64).reshape(4)
    norm = np.linalg.norm((x, y, z, w))
    if norm == 0:
        raise ValueError("Camera quaternion has zero norm")
    x, y, z, w = np.asarray((x, y, z, w)) / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def orientation_error_deg(actual: np.ndarray, expected: np.ndarray) -> float:
    """Smallest angular difference between two column-vector rotation matrices."""
    return rotation_angle_deg(actual.T @ expected)


def minimum_surface_distance_mm(points_a: np.ndarray, points_b: np.ndarray) -> dict[str, float | int]:
    deltas = points_a[:, None, :] - points_b[None, :, :]
    distances = np.linalg.norm(deltas, axis=-1)
    return {
        "minimum_mm": float(distances.min() * 1000.0),
        "pairs_within_0_5mm": int(np.count_nonzero(distances <= 0.0005)),
        "pairs_within_1mm": int(np.count_nonzero(distances <= 0.001)),
    }


def local_points(points_world: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Undo the row-vector initial-to-current moving-frame transform."""
    return (points_world - translation) @ rotation.T


def attachment_target_error(item: dict[str, np.ndarray]) -> dict[str, float | int] | None:
    """Measure the constrained gel nodes against the rigid-body target positions."""
    required = {"gelpad_vertices", "attachment_point_indices", "attachment_targets"}
    if not required.issubset(item):
        return None
    indices = np.asarray(item["attachment_point_indices"], dtype=np.int64).reshape(-1)
    vertices = np.asarray(item["gelpad_vertices"], dtype=np.float64).reshape(-1, 3)
    targets = np.asarray(item["attachment_targets"], dtype=np.float64).reshape(-1, 3)
    if len(indices) != len(targets):
        raise ValueError(f"Attachment index/target count mismatch: {len(indices)} != {len(targets)}")
    if np.any(indices < 0) or np.any(indices >= len(vertices)):
        raise ValueError(f"Attachment indices are outside the {len(vertices)} gel vertices")
    error_mm = np.linalg.norm(vertices[indices] - targets, axis=-1) * 1000.0
    return {
        "count": int(len(indices)),
        "mean_mm": float(error_mm.mean()),
        "max_mm": float(error_mm.max()),
        "p95_mm": float(np.percentile(error_mm, 95)),
    }


def attachment_vertices(item: dict[str, np.ndarray]) -> np.ndarray | None:
    if "attachment_point_indices" not in item or "gelpad_vertices" not in item:
        return None
    indices = np.asarray(item["attachment_point_indices"], dtype=np.int64).reshape(-1)
    vertices = np.asarray(item["gelpad_vertices"], dtype=np.float64).reshape(-1, 3)
    return vertices[indices]


def rigid_motion_summary(before: np.ndarray, after: np.ndarray) -> dict[str, float]:
    """Best row-vector rigid motion from matching point sets, for diagnostics."""
    before_center = before.mean(axis=0)
    after_center = after.mean(axis=0)
    covariance = (before - before_center).T @ (after - after_center)
    left, _, right_t = np.linalg.svd(covariance)
    rotation = left @ right_t
    if np.linalg.det(rotation) < 0:
        right_t[-1] *= -1
        rotation = left @ right_t
    translation = after_center - before_center @ rotation
    fit_error = before @ rotation + translation - after
    return {
        "rotation_angle_deg": rotation_angle_deg(rotation.T),
        "translation_norm_mm": float(np.linalg.norm(translation) * 1000.0),
        "fit_rms_mm": float(np.sqrt(np.mean(np.sum(fit_error**2, axis=-1))) * 1000.0),
    }


def snapshot(root: Path, label: str, side: str) -> dict[str, np.ndarray]:
    directory = root / label
    prefix = f"{side}_"
    result = {
        "camera_pos_w": load(directory / f"{prefix}camera_pos_w.npy").reshape(-1, 3)[0],
        "camera_quat_w_ros": load(directory / f"{prefix}camera_quat_w_ros.npy").reshape(-1, 4)[0],
        "raw_rgb": load(directory / f"{prefix}camera_rgb_raw.npy"),
        "raw_depth": load(directory / f"{prefix}camera_depth_raw.npy"),
        "tactile_rgb": load(directory / f"{prefix}rgb.npy"),
        "marker_rgb": load(directory / f"{prefix}marker_rgb.npy"),
        "height_map": load(directory / f"{prefix}depth.npy"),
        "surface_world": load(directory / f"{prefix}gelpad_surface_vertices_world.npy"),
        "rotation": load(directory / f"{prefix}moving_frame_rotation_world_from_initial.npy"),
        "translation": load(directory / f"{prefix}moving_frame_translation_world_from_initial.npy"),
        "fit_rms_m": load(directory / f"{prefix}marker_rigid_fit_rms_m.npy").item(),
    }
    initial_camera_path = directory / f"{prefix}initial_camera_pos_w.npy"
    if initial_camera_path.exists():
        result["initial_camera_pos_w"] = load(initial_camera_path).reshape(-1, 3)[0]
        result["initial_camera_quat_w_ros"] = load(
            directory / f"{prefix}initial_camera_quat_w_ros.npy"
        ).reshape(-1, 4)[0]
    attachment_indices = directory / f"{prefix}attachment_point_indices.npy"
    if attachment_indices.exists():
        result["attachment_point_indices"] = load(attachment_indices)
        result["attachment_targets"] = load(directory / f"{prefix}attachment_targets.npy")
        result["gelpad_vertices"] = load(directory / f"{prefix}gelpad_vertices.npy")
    return result


def build_report(root: Path, labels: list[str], initial_camera_reference: Path | None = None) -> dict:
    available = [label for label in labels if (root / label).is_dir()]
    if len(available) < 2:
        raise ValueError(f"Need at least two snapshot directories, found: {available}")

    report: dict = {"snapshots": available, "sides": {}}
    prism_path = root / available[0] / "actor_prism.npy"
    prism_by_label = {label: load(root / label / "actor_prism.npy") for label in available} if prism_path.exists() else {}
    actor_names = sorted(path.stem.removeprefix("actor_") for path in (root / available[0]).glob("actor_*.npy"))
    actors_by_label = {
        label: {name: load(root / label / f"actor_{name}.npy") for name in actor_names}
        for label in available
    }

    for side in ("left_tactile", "right_tactile"):
        samples = {label: snapshot(root, label, side) for label in available}
        first_label = available[0]
        first = samples[first_label]
        initial_camera_position = first.get("initial_camera_pos_w")
        initial_camera_orientation = first.get("initial_camera_quat_w_ros")
        if initial_camera_position is None and initial_camera_reference is not None:
            reference_path = initial_camera_reference / "t0" / f"{side}_camera_pos_w.npy"
            if not reference_path.exists():
                raise FileNotFoundError(f"Initial camera reference is missing: {reference_path}")
            initial_camera_position = load(reference_path).reshape(-1, 3)[0]
            initial_camera_orientation = load(
                initial_camera_reference / "t0" / f"{side}_camera_quat_w_ros.npy"
            ).reshape(-1, 4)[0]
        side_report = {
            "per_snapshot": {},
            "comparisons_from_first": {},
        }
        for label, item in samples.items():
            local_surface = local_points(item["surface_world"], item["rotation"], item["translation"])
            entry = {
                "reported_camera_pos_w_m": item["camera_pos_w"].tolist(),
                "moving_frame_translation_m": item["translation"].tolist(),
                "moving_frame_rotation_angle_deg": rotation_angle_deg(item["rotation"]),
                "rigid_fit_rms_m": float(item["fit_rms_m"]),
                "height_map": center_depth_stats(item["height_map"]),
                "surface_local_centroid_m": local_surface.mean(axis=0).tolist(),
            }
            attachment_error = attachment_target_error(item)
            if attachment_error is not None:
                entry["attachment_target_error"] = attachment_error
            if initial_camera_position is not None:
                expected_camera_position = initial_camera_position @ item["rotation"] + item["translation"]
                entry["expected_camera_pos_w_m"] = expected_camera_position.tolist()
                entry["camera_pose_error_norm_m"] = float(
                    np.linalg.norm(item["camera_pos_w"] - expected_camera_position)
                )
            if initial_camera_orientation is not None:
                expected_camera_orientation = item["rotation"].T @ rotation_matrix_from_quat_xyzw(
                    initial_camera_orientation
                )
                entry["camera_orientation_error_deg"] = orientation_error_deg(
                    rotation_matrix_from_quat_xyzw(item["camera_quat_w_ros"]), expected_camera_orientation
                )
            if label in prism_by_label:
                prism_local = local_points(prism_by_label[label], item["rotation"], item["translation"])
                entry["prism_local_centroid_m"] = prism_local.mean(axis=0).tolist()
                entry["surface_proximity_to_prism"] = minimum_surface_distance_mm(
                    item["surface_world"], prism_by_label[label]
                )
            if actors_by_label[label]:
                entry["surface_proximity_to_actors"] = {
                    name: minimum_surface_distance_mm(item["surface_world"], vertices)
                    for name, vertices in actors_by_label[label].items()
                }
            side_report["per_snapshot"][label] = entry

        for label in available[1:]:
            item = samples[label]
            # If the embedded camera were attached to the moving gelpad, this
            # is the world-space displacement implied by the reconstructed
            # UIPC frame.  The reported camera displacement is measured from
            # Isaac Lab's nested camera view.
            camera_reference = initial_camera_position if initial_camera_position is not None else first["camera_pos_w"]
            expected_first_camera_position = camera_reference @ first["rotation"] + first["translation"]
            expected_camera_position = camera_reference @ item["rotation"] + item["translation"]
            expected_camera_delta = expected_camera_position - expected_first_camera_position
            reported_camera_delta = item["camera_pos_w"] - first["camera_pos_w"]
            first_surface_local = local_points(first["surface_world"], first["rotation"], first["translation"])
            current_surface_local = local_points(item["surface_world"], item["rotation"], item["translation"])
            comparison = {
                "reported_camera_delta_norm_m": float(np.linalg.norm(reported_camera_delta)),
                "expected_camera_delta_norm_m": float(np.linalg.norm(expected_camera_delta)),
                "raw_camera_rgb": image_metrics(first["raw_rgb"], item["raw_rgb"]),
                "raw_camera_depth": image_metrics(first["raw_depth"], item["raw_depth"]),
                "tactile_rgb": image_metrics(first["tactile_rgb"], item["tactile_rgb"]),
                "marker_rgb": image_metrics(first["marker_rgb"], item["marker_rgb"]),
                "height_map": image_metrics(first["height_map"], item["height_map"]),
                "gel_surface_local_displacement_mm": {
                    "mean": float(np.linalg.norm(current_surface_local - first_surface_local, axis=-1).mean() * 1000.0),
                    "max": float(np.linalg.norm(current_surface_local - first_surface_local, axis=-1).max() * 1000.0),
                },
            }
            if label in prism_by_label and first_label in prism_by_label:
                first_prism_local = local_points(prism_by_label[first_label], first["rotation"], first["translation"])
                current_prism_local = local_points(prism_by_label[label], item["rotation"], item["translation"])
                comparison["prism_local_displacement_mm"] = {
                    "mean": float(np.linalg.norm(current_prism_local - first_prism_local, axis=-1).mean() * 1000.0),
                    "max": float(np.linalg.norm(current_prism_local - first_prism_local, axis=-1).max() * 1000.0),
                }
            if initial_camera_position is not None:
                comparison["first_camera_pose_error_norm_m"] = float(
                    np.linalg.norm(first["camera_pos_w"] - expected_first_camera_position)
                )
                comparison["camera_pose_error_norm_m"] = float(
                    np.linalg.norm(item["camera_pos_w"] - expected_camera_position)
                )
            if initial_camera_orientation is not None:
                initial_orientation_matrix = rotation_matrix_from_quat_xyzw(initial_camera_orientation)
                expected_first_orientation = first["rotation"].T @ initial_orientation_matrix
                expected_orientation = item["rotation"].T @ initial_orientation_matrix
                comparison["first_camera_orientation_error_deg"] = orientation_error_deg(
                    rotation_matrix_from_quat_xyzw(first["camera_quat_w_ros"]), expected_first_orientation
                )
                comparison["camera_orientation_error_deg"] = orientation_error_deg(
                    rotation_matrix_from_quat_xyzw(item["camera_quat_w_ros"]), expected_orientation
                )
            first_attachment_vertices = attachment_vertices(first)
            current_attachment_vertices = attachment_vertices(item)
            if first_attachment_vertices is not None and current_attachment_vertices is not None:
                first_targets = np.asarray(first["attachment_targets"], dtype=np.float64).reshape(-1, 3)
                current_targets = np.asarray(item["attachment_targets"], dtype=np.float64).reshape(-1, 3)
                target_displacement = current_targets - first_targets
                vertex_displacement = current_attachment_vertices - first_attachment_vertices
                mismatch_mm = np.linalg.norm(vertex_displacement - target_displacement, axis=-1) * 1000.0
                comparison["attachment_motion"] = {
                    "target": rigid_motion_summary(first_targets, current_targets),
                    "constrained_vertices": rigid_motion_summary(first_attachment_vertices, current_attachment_vertices),
                    "vertex_minus_target_displacement_mm": {
                        "mean": float(mismatch_mm.mean()),
                        "max": float(mismatch_mm.max()),
                        "p95": float(np.percentile(mismatch_mm, 95)),
                    },
                }
            side_report["comparisons_from_first"][label] = comparison
        report["sides"][side] = side_report
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("validation_dir", type=Path)
    parser.add_argument(
        "--labels",
        nargs="+",
        default=["bilateral_contact_start", "bilateral_contact_motion", "left_contact_loss"],
        help="Snapshot labels in chronological order.",
    )
    parser.add_argument(
        "--initial-camera-reference",
        type=Path,
        default=None,
        help="Earlier probe directory whose t0 camera pose is an immutable initial-frame reference.",
    )
    args = parser.parse_args()
    report = build_report(args.validation_dir, args.labels, args.initial_camera_reference)
    output = args.validation_dir / "frame_consistency_report.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
