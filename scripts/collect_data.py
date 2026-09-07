import os
import sys
import time
import yaml
import json
import torch
import argparse
import traceback
import numpy as np
from pathlib import Path
from typing import TYPE_CHECKING, Literal

sys.path.append('.')

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Collect data"
)
parser.add_argument(
    "task",
    type=str,
    help="Task file name",
)
parser.add_argument(
    "config",
    type=str,
    help="Config file name",
    default="demo.yml"
)
parser.add_argument(
    "--episode_num",
    type=int,
    default=-1,
)
parser.add_argument(
    "--start_seed",
    type=int,
    default=-1,
)
parser.add_argument(
    "--max_seed",
    type=int,
    default=-1,
)
parser.add_argument(
    "--gpu",
    type=str,
    default=None,
)
parser.add_argument(
    "--validation-dir",
    type=Path,
    default=None,
    help="Write opt-in raw sensor/state snapshots for a bounded runtime validation.",
)
parser.add_argument(
    "--validation-marker-probe",
    action="store_true",
    help="After reset, capture marker-projection diagnostics and exit without playing an episode.",
)
parser.add_argument(
    "--validation-tactile-sanity",
    action="store_true",
    help="Run an opt-in no-contact / bilateral-contact / release observation sanity sequence and exit.",
)

from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)

# Parse only after registering the Isaac Lab flags. This makes the standard
# ``--headless --livestream 0`` Docker invocation usable instead of silently
# forcing the full GUI/streaming renderer on a memory-constrained workstation.
args_cli = parser.parse_args()
if args_cli.gpu is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = args_cli.gpu

args_cli.enable_cameras = True
args_cli.num_envs = 1

def get_config(file, default_root:Path, type:Literal['yaml', 'json']):
    if type == 'yaml':
        if file.endswith('.yml') or file.endswith('.yaml'):
            file = Path(file)
        else:
            file = default_root / f'{file}.yml'
        with open(file, 'r') as f:
            config = yaml.load(f.read(), Loader=yaml.FullLoader)
        return config, file
    else:
        if file.endswith('.json'):
            file = Path(file)
        else:
            file = default_root / f'{file}.json'
        with open(file, 'r') as f:
            config = json.load(f)
        return config, file

task_config, task_config_file = get_config(
    args_cli.config, 
    default_root=Path(__file__).parent.parent / 'task_config', 
    type='yaml'
)

# ``render_frequency`` controls when this task renders its cameras; it is not
# a request to start WebRTC. Streaming starts the non-headless rendering
# experience and consumes resources that are unnecessary for collection.

# launch omniverse app, must done before importing anything from omni.isaac
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib
if TYPE_CHECKING:
    from envs._base_task import BaseTask, BaseTaskCfg

log_path = Path('./log')
def log(msg):
    global log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)

    msg = f"[{time.strftime(r'%Y-%m-%d %H:%M:%S')}] {msg}"
    with open(log_path, 'a') as f:
        f.write(msg + '\n')
    print(msg)


class ValidationCapture:
    """Read-only capture of a task's settled, pre-contact, and post-contact states.

    This deliberately wraps the scripted task at runtime instead of changing task logic,
    PhysX, UIPC, camera, or tactile settings.  It is enabled only by
    ``--validation-dir`` and is intended for a one-episode hardware/runtime check.
    """

    def __init__(self, task: 'BaseTask', output_dir: Path):
        self.task = task
        self.output_dir = output_dir
        self.snapshots: dict[str, dict] = {}
        self.pre_move_calls = 0
        self.pre_contact_captured = False
        self.post_contact_captured = False
        self.capture_no_contact = False
        self.bilateral_contact_start_step = None
        self.bilateral_contact_motion_captured = False
        self.left_contact_loss_captured = False
        self.contact_trace: list[dict] = []
        self.plan_trace: list[dict] = []
        self._previous_attachment_targets: dict[str, np.ndarray] = {}
        self.plan_failure_recorded = False
        self._install()

    @staticmethod
    def _to_numpy(value):
        if hasattr(value, "torch"):
            value = value.torch
        elif value.__class__.__module__.startswith("warp"):
            import warp as wp

            value = wp.to_torch(value)
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    @staticmethod
    def _array_summary(value):
        array = ValidationCapture._to_numpy(value)
        finite = np.isfinite(array)
        finite_values = array[finite]
        summary = {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "count": int(array.size),
            "finite_count": int(finite.sum()),
            "nan_count": int(np.isnan(array).sum()),
            "inf_count": int(np.isinf(array).sum()),
            "finite_fraction": float(finite.mean()) if array.size else 1.0,
        }
        if finite_values.size:
            summary.update({
                "min": float(finite_values.min()),
                "max": float(finite_values.max()),
                "mean": float(finite_values.mean()),
                "std": float(finite_values.std()),
            })
        else:
            summary.update({"min": None, "max": None, "mean": None, "std": None})
        return summary

    @staticmethod
    def _depth_pipeline_summary(tactile, arrays, name: str):
        """Describe the read-only RTX-to-height-map observation path in millimetres."""
        height_map_mm = np.squeeze(arrays[f"tactile/{name}/depth"]).astype(np.float64)
        annotator_m = np.squeeze(arrays[f"tactile/{name}/camera_depth_annotator"]).astype(np.float64)
        raw_camera_m = np.squeeze(arrays[f"tactile/{name}/camera_depth_raw"]).astype(np.float64)

        def depth_stats_mm(values_m):
            values_mm = values_m * 1000.0
            finite = values_mm[np.isfinite(values_mm)]
            return {
                "finite_fraction": float(np.isfinite(values_mm).mean()),
                "min_mm": None if not finite.size else float(finite.min()),
                "max_mm": None if not finite.size else float(finite.max()),
                "near_20mm_fraction": float(np.mean(np.abs(values_mm - 20.0) <= 0.1)),
            }

        shared_finite = np.isfinite(annotator_m) & np.isfinite(height_map_mm)
        if shared_finite.any():
            annotator_to_height_error_mm = np.abs(height_map_mm[shared_finite] - annotator_m[shared_finite] * 1000.0)
            annotator_to_height = {
                "shared_finite_fraction": float(shared_finite.mean()),
                "mean_absolute_error_mm": float(annotator_to_height_error_mm.mean()),
                "max_absolute_error_mm": float(annotator_to_height_error_mm.max()),
            }
        else:
            annotator_to_height = {
                "shared_finite_fraction": 0.0,
                "mean_absolute_error_mm": None,
                "max_absolute_error_mm": None,
            }
        height_finite = height_map_mm[np.isfinite(height_map_mm)]
        return {
            "configured_clipping_range_m": [float(value) for value in tactile.sensor.cfg.sensor_camera_cfg.clipping_range],
            "authored_camera_planes": tactile.sensor.get_camera_plane_diagnostics(),
            "renderer_annotator_m": depth_stats_mm(annotator_m),
            "tiled_camera_buffer_m": depth_stats_mm(raw_camera_m),
            "height_map_mm": {
                "min_mm": None if not height_finite.size else float(height_finite.min()),
                "max_mm": None if not height_finite.size else float(height_finite.max()),
                "near_20mm_fraction": float(np.mean(np.abs(height_map_mm - 20.0) <= 0.1)),
            },
            "height_map_minus_finite_annotator": annotator_to_height,
        }

    @staticmethod
    def _save_rgb(path: Path, value):
        from PIL import Image

        image = ValidationCapture._to_numpy(value)
        if image.ndim == 3 and image.shape[-1] == 4:
            image = image[..., :3]
        if image.dtype != np.uint8:
            upper = 1.0 if image.size and np.nanmax(image) <= 1.0 else 255.0
            image = np.clip(image * (255.0 / upper), 0, 255).astype(np.uint8)
        Image.fromarray(image).save(path)

    def _snapshot_arrays(self, include_uipc_surface: bool = False):
        arrays = {}
        camera_observations = self.task._camera_manager.get_observations(["rgb", "depth"])
        for name, camera in self.task._camera_manager.cameras.items():
            arrays[f"camera/{name}/rgb"] = camera.data.output["rgb"].squeeze(0)
            arrays[f"camera/{name}/depth"] = camera.data.output["depth"].squeeze(0)
            arrays[f"camera/{name}/observation_rgb"] = camera_observations[name]["rgb"]
            arrays[f"camera/{name}/observation_depth"] = camera_observations[name]["depth"]
            # Regular scene cameras are captured alongside tactile data so an
            # invalid RGB frame can be localized to a pose/intrinsics issue,
            # the RTX annotator, or Isaac Lab's tiled-camera buffer.  These
            # reads use buffers already updated for this frame and do not
            # trigger another render or alter task state.
            arrays[f"camera/{name}/pos_w"] = camera.data.pos_w
            arrays[f"camera/{name}/quat_w_ros"] = camera.data.quat_w_ros
            arrays[f"camera/{name}/quat_w_opengl"] = camera.data.quat_w_opengl
            arrays[f"camera/{name}/intrinsic_matrices"] = camera.data.intrinsic_matrices
            render_data = getattr(camera, "_render_data", None)
            rgb_annotator = None if render_data is None else render_data.annotators.get("rgb")
            if rgb_annotator is not None:
                rgb_payload = rgb_annotator.get_data()
                if isinstance(rgb_payload, dict):
                    rgb_payload = rgb_payload["data"]
                arrays[f"camera/{name}/rgb_annotator"] = rgb_payload
        for name, tactile in self.task._tactile_manager.tactiles.items():
            arrays[f"tactile/{name}/rgb"] = tactile.sensor.data.output["tactile_rgb"].squeeze(0)
            arrays[f"tactile/{name}/marker_rgb"] = tactile.sensor.data.output["marker_rgb"].squeeze(0)
            arrays[f"tactile/{name}/camera_rgb_raw"] = tactile.sensor.camera.data.output["rgb"].squeeze(0)
            # These are cached products of the marker simulation that already ran
            # for this frame.  Capturing them is diagnostic-only: it does not
            # trigger a simulation, renderer update, or camera readback.
            arrays[f"tactile/{name}/marker_motion"] = tactile.sensor.data.output["marker_motion"].squeeze(0)
            marker_simulator = tactile.sensor.marker_motion_simulator
            if marker_simulator is not None:
                marker_sim = marker_simulator.marker_motion_sim
                if hasattr(marker_sim, "curr_marker_uv"):
                    arrays[f"tactile/{name}/marker_uv_raw"] = marker_sim.curr_marker_uv
                arrays[f"tactile/{name}/marker_grid"] = marker_sim.marker_grid
                arrays[f"tactile/{name}/marker_reference_surface_camera"] = (
                    marker_sim.reference_surface_vertices_camera
                )
                arrays[f"tactile/{name}/marker_camera_intrinsic"] = marker_sim.camera_intrinsic
                # Immutable pre-correction pose used by the marker model.  It
                # lets the offline validator check the RTX camera pose directly
                # against the UIPC moving frame without treating a dynamic
                # first snapshot as a static reference.
                arrays[f"tactile/{name}/initial_camera_pos_w"] = marker_sim.initial_camera_pos_w
                arrays[f"tactile/{name}/initial_camera_quat_w_ros"] = marker_sim.initial_camera_quat_w_ros
                for label, attribute in (
                    ("marker_initial_points_camera", "last_marker_initial_points_camera"),
                    ("marker_current_points_camera", "last_marker_current_points_camera"),
                    ("marker_in_bounds", "last_marker_in_bounds"),
                    ("marker_rigid_fit_rms_m", "last_marker_rigid_fit_rms_m"),
                    ("marker_rigid_fit_max_m", "last_marker_rigid_fit_max_m"),
                    ("moving_frame_rotation_world_from_initial", "last_gelpad_rotation_world_from_initial"),
                    ("moving_frame_translation_world_from_initial", "last_gelpad_translation_world_from_initial"),
                ):
                    if hasattr(marker_sim, attribute):
                        arrays[f"tactile/{name}/{label}"] = getattr(marker_sim, attribute)
            arrays[f"tactile/{name}/camera_pos_w"] = tactile.sensor.camera.data.pos_w
            arrays[f"tactile/{name}/camera_quat_w_ros"] = tactile.sensor.camera.data.quat_w_ros
            arrays[f"tactile/{name}/depth"] = tactile.sensor.data.output["height_map"].squeeze(0)
            arrays[f"tactile/{name}/camera_depth_raw"] = tactile.sensor.camera.data.output["depth"].squeeze(0)
            # Preserve the RTX annotator payload as well as Isaac Lab's reshaped
            # camera buffer.  The opt-in probe uses this only to localize a
            # renderer-versus-buffer failure; it does not affect tactile output.
            render_data = tactile.sensor.camera._render_data
            depth_annotator = render_data.annotators.get("depth")
            if depth_annotator is not None:
                depth_payload = depth_annotator.get_data()
                if isinstance(depth_payload, dict):
                    depth_payload = depth_payload["data"]
                arrays[f"tactile/{name}/camera_depth_annotator"] = depth_payload
            # These are the two inputs to the tactile pipeline: the UIPC
            # deformable mesh and the fixed attachment targets supplied by
            # Isaac.  They are captured only for the opt-in runtime probe.
            arrays[f"tactile/{name}/gelpad_vertices"] = tactile.gelpad.data.nodal_pos_w
            if marker_simulator is not None:
                arrays[f"tactile/{name}/gelpad_surface_vertices_world"] = marker_sim.get_surface_vertices_world()
            arrays[f"tactile/{name}/attachment_point_indices"] = tactile.attachment.attachment_points_idx
            arrays[f"tactile/{name}/attachment_offsets"] = tactile.attachment.attachment_offsets
            arrays[f"tactile/{name}/attachment_targets"] = tactile.attachment.aim_positions
            if tactile.attachment.obj_pose is not None:
                arrays[f"tactile/{name}/attachment_rigid_pose"] = tactile.attachment.obj_pose
        for name, actor in self.task._actor_manager.actors.items():
            arrays[f"actor/{name}/surface_vertices_world"] = actor.vertices
        if include_uipc_surface:
            arrays["uipc/surface_vertices"] = self.task.uipc_sim.sio.simplicial_surface(2).positions().view()
        robot = self.task._robot_manager.robot.data
        arrays["robot/joint_pos"] = robot.joint_pos.squeeze(0)
        arrays["robot/joint_vel"] = robot.joint_vel.squeeze(0)
        arrays["robot/ee_pose"] = self.task._robot_manager.get_ee_pose().totensor()
        return {name: self._to_numpy(value) for name, value in arrays.items()}

    def _save_arrays(self, label: str, arrays: dict[str, np.ndarray]):
        label_dir = self.output_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)
        for name, value in arrays.items():
            parts = name.split("/")
            family, sensor = parts[:2]
            data_type = parts[2] if len(parts) == 3 else None
            if family == "camera":
                stem = f"{sensor}_{data_type}"
            elif family == "tactile":
                stem = f"{sensor}_{data_type}"
            else:
                stem = f"{family}_{sensor}"
            np.save(label_dir / f"{stem}.npy", value)
            if data_type in {"rgb", "marker_rgb", "observation_rgb"}:
                self._save_rgb(label_dir / f"{stem}.png", value)

    def capture(self, label: str):
        if label in self.snapshots:
            return
        arrays = self._snapshot_arrays(include_uipc_surface=label == "failure")
        self._save_arrays(label, arrays)
        elapsed = time.perf_counter() - self.task.start_time
        actor_poses = {
            name: np.concatenate((actor.get_pose().p, actor.get_pose().q))
            for name, actor in self.task._actor_manager.actors.items()
        }
        previous = list(self.snapshots.values())[-1] if self.snapshots else None
        actor_velocities = {}
        if previous is not None:
            dt = elapsed - previous["elapsed_s"]
            if dt > 0:
                for name, pose in actor_poses.items():
                    if name in previous["actor_poses"]:
                        actor_velocities[name] = ((pose[:3] - previous["actor_poses"][name][:3]) / dt).tolist()
        tactile_attachment_counts = {
            name: int(len(tactile.attachment.attachment_points_idx))
            for name, tactile in self.task._tactile_manager.tactiles.items()
        }
        tactile_attachment_diagnostics = {
            name: tactile.attachment.get_diagnostics()
            for name, tactile in self.task._tactile_manager.tactiles.items()
        }
        tactile_depth_pipeline = {
            name: self._depth_pipeline_summary(tactile, arrays, name)
            for name, tactile in self.task._tactile_manager.tactiles.items()
        }
        state = {
            "label": label,
            "step_count": int(self.task.step_count),
            "elapsed_s": float(elapsed),
            "array_stats": {name: self._array_summary(value) for name, value in arrays.items()},
            "object_pose_xyz_xyzw": {name: pose.tolist() for name, pose in actor_poses.items()},
            # Actor poses originate from ``Pose``/transforms3d (WXYZ), unlike
            # attachment rigid poses which are read from PhysX (XYZW).
            # Preserve the legacy key for existing artifacts but make its
            # convention explicit for valid reference comparisons.
            "object_pose_quaternion_convention": "wxyz",
            "object_linear_velocity_m_per_s_from_capture_delta": actor_velocities,
            "tactile_attachment_counts": tactile_attachment_counts,
            "tactile_attachment_diagnostics": tactile_attachment_diagnostics,
            "tactile_depth_pipeline": tactile_depth_pipeline,
            "camera_attachment_diagnostics": self.task._camera_manager.get_attachment_diagnostics(),
            "tactile_min_depth": {
                name: float(np.nanmin(arrays[f"tactile/{name}/depth"]))
                for name in self.task._tactile_manager.tactiles
            },
        }
        with open(self.output_dir / f"{label}_state.json", "w", encoding="utf-8") as file:
            json.dump(state, file, indent=2, allow_nan=False)
        self.snapshots[label] = {
            "elapsed_s": elapsed,
            "actor_poses": actor_poses,
            "arrays": arrays,
            "state": state,
        }
        log(f"Validation snapshot {label} saved to {self.output_dir}")

    def _active_actor_surface_gap_mm(self, tactile):
        """Return a read-only, unsigned pad-to-active-object surface gap.

        The value is the exact nearest distance between the marker simulator's
        current gel surface vertices and the active task actor's current UIPC
        surface vertices.  It is deliberately an unsigned point-cloud metric:
        it can distinguish a millimetre-scale physical miss from near-contact,
        but it does not claim to be a signed collision distance or alter either
        simulation.  Chunking bounds diagnostic memory without subsampling.
        """
        active_actor = getattr(self.task, "can", None)
        marker_simulator = getattr(tactile.sensor, "marker_motion_simulator", None)
        if active_actor is None or marker_simulator is None:
            return None

        gel_surface = self._to_numpy(
            marker_simulator.marker_motion_sim.get_surface_vertices_world()
        ).reshape(-1, 3)
        actor_surface = self._to_numpy(active_actor.vertices).reshape(-1, 3)
        gel_surface = gel_surface[np.all(np.isfinite(gel_surface), axis=1)]
        actor_surface = actor_surface[np.all(np.isfinite(actor_surface), axis=1)]
        if not len(gel_surface) or not len(actor_surface):
            return None

        nearest_sq = np.full(len(gel_surface), np.inf, dtype=np.float64)
        nearest_actor_indices = np.full(len(gel_surface), -1, dtype=np.int64)
        for start in range(0, len(actor_surface), 4096):
            actor_chunk = actor_surface[start : start + 4096]
            offset = gel_surface[:, None, :] - actor_chunk[None, :, :]
            squared_distance = np.einsum("...i,...i->...", offset, offset)
            local_indices = np.argmin(squared_distance, axis=1)
            local_nearest_sq = squared_distance[np.arange(len(gel_surface)), local_indices]
            update = local_nearest_sq < nearest_sq
            nearest_sq[update] = local_nearest_sq[update]
            nearest_actor_indices[update] = start + local_indices[update]

        nearest_mm = np.sqrt(nearest_sq) * 1000.0
        closest_gel_index = int(np.argmin(nearest_mm))
        closest_actor_index = int(nearest_actor_indices[closest_gel_index])
        return {
            "actor": active_actor.cfg.name,
            "gel_surface_vertex_count": int(len(gel_surface)),
            "actor_surface_vertex_count": int(len(actor_surface)),
            "min_mm": float(nearest_mm[closest_gel_index]),
            "p05_mm": float(np.percentile(nearest_mm, 5)),
            "median_mm": float(np.median(nearest_mm)),
            "closest_gel_surface_vertex_world_m": gel_surface[closest_gel_index].tolist(),
            "closest_actor_surface_vertex_world_m": actor_surface[closest_actor_index].tolist(),
        }

    def _record_contact_trace(self):
        """Record the already-rendered contact state for a saved collect frame.

        This intentionally reads the same buffers that ``_step`` just saved.  It
        neither advances simulation nor requests an additional render, keeping
        the trace diagnostic-only for a one-episode validation run.
        """
        if self.task.in_pre_move or self.task.step_count % self.task.cfg.save_frequency:
            return

        tactile_state = {}
        for name, tactile in self.task._tactile_manager.tactiles.items():
            depth = self._to_numpy(tactile.sensor.data.output["height_map"].squeeze(0))
            center = depth[50:-50, 50:-50]
            attachment_indices = np.asarray(tactile.attachment.attachment_points_idx, dtype=np.int64).reshape(-1)
            gel_vertices = self._to_numpy(tactile.gelpad.data.nodal_pos_w).reshape(-1, 3)
            attachment_targets = np.asarray(tactile.attachment.aim_positions, dtype=np.float64).reshape(-1, 3)
            attachment_error_mm = np.linalg.norm(
                gel_vertices[attachment_indices] - attachment_targets, axis=-1
            ) * 1000.0
            previous_targets = self._previous_attachment_targets.get(name)
            if previous_targets is None or previous_targets.shape != attachment_targets.shape:
                target_displacement_mm = None
            else:
                target_displacement_mm = np.linalg.norm(attachment_targets - previous_targets, axis=-1) * 1000.0
            self._previous_attachment_targets[name] = attachment_targets.copy()
            attachment_diagnostics = tactile.attachment.get_diagnostics()
            attachment_rigid_pose = None
            if tactile.attachment.obj_pose is not None:
                attachment_rigid_pose = self._to_numpy(tactile.attachment.obj_pose).reshape(-1, 7)[0].tolist()
            tactile_state[name] = {
                "min_depth_mm": float(np.min(depth)),
                "center_min_depth_mm": float(np.min(center)),
                "center_max_depth_mm": float(np.max(center)),
                "center_has_contact": bool(np.min(center) != np.max(center)),
                "attachment_target_error_mm": {
                    "mean": float(attachment_error_mm.mean()),
                    "max": float(attachment_error_mm.max()),
                    "p95": float(np.percentile(attachment_error_mm, 95)),
                },
                "attachment_rigid_pose_xyz_xyzw": attachment_rigid_pose,
                "attachment_target_displacement_mm_from_previous_saved_frame": None
                if target_displacement_mm is None
                else {
                    "mean": float(target_displacement_mm.mean()),
                    "max": float(target_displacement_mm.max()),
                    "p95": float(np.percentile(target_displacement_mm, 95)),
                },
                "attachment_callback_generation_lag": int(
                    attachment_diagnostics["aim_generation_minus_last_animation"]
                ),
                "attachment_uipc_callback_order": {
                    "uipc_step_serial_minus_last_aim": int(
                        attachment_diagnostics["uipc_step_serial_minus_last_aim"]
                    ),
                    "last_aim_uipc_step_phase": attachment_diagnostics["last_aim_uipc_step_phase"],
                    "last_animation_uipc_step_phase": attachment_diagnostics[
                        "last_animation_uipc_step_phase"
                    ],
                },
                # Atom 1 is the diagnostic boundary: record whether either
                # pad actually approaches the active can before the adaptive
                # closure decides to continue.  This reads existing UIPC
                # buffers only; it does not request a render or a simulation
                # step.
                "active_actor_surface_gap_mm": self._active_actor_surface_gap_mm(tactile)
                if self.task.atom_id == 1
                else None,
            }

        # Capture the articulation state and command buffer at the same saved
        # frame as the gel attachment measurements.  This distinguishes an
        # attachment/physics loss from an upstream controller-tracking loss,
        # without issuing another command or stepping the simulator.
        robot_data = self.task._robot_manager.robot.data
        joint_pos = self._to_numpy(robot_data.joint_pos.squeeze(0))
        joint_vel = self._to_numpy(robot_data.joint_vel.squeeze(0))
        joint_pos_target = self._to_numpy(robot_data.joint_pos_target.squeeze(0))
        joint_vel_target = self._to_numpy(robot_data.joint_vel_target.squeeze(0))
        arm_joint_ids = self._to_numpy(self.task._robot_manager._arm_ids).astype(np.int64)
        arm_position_target_error = joint_pos_target[arm_joint_ids] - joint_pos[arm_joint_ids]
        actuator_effort = {}
        for name, actuator in self.task._robot_manager.robot.actuators.items():
            effort = {}
            for attribute in ("computed_effort", "applied_effort"):
                value = getattr(actuator, attribute, None)
                if value is not None:
                    effort[attribute] = self._to_numpy(value).tolist()
            if effort:
                actuator_effort[name] = effort

        trace_entry = (
            {
                "step_count": int(self.task.step_count),
                "atom_id": int(self.task.atom_id),
                "atom_tag": self.task.atom_tag,
                "gripper_qpos": float(self.task._robot_manager.get_gripper_qpos()),
                "robot_joint_state": {
                    "position": joint_pos.tolist(),
                    "velocity": joint_vel.tolist(),
                    "position_target": joint_pos_target.tolist(),
                    "velocity_target": joint_vel_target.tolist(),
                    "arm_position_target_error": arm_position_target_error.tolist(),
                    "arm_position_target_error_norm": float(np.linalg.norm(arm_position_target_error)),
                    "arm_position_target_error_max_abs": float(np.max(np.abs(arm_position_target_error))),
                    "actuator_effort": actuator_effort,
                },
                "ee_pose": self.task._robot_manager.get_ee_pose().totensor().tolist(),
                "actor_pose_xyz_xyzw": {
                    name: np.concatenate((actor.get_pose().p, actor.get_pose().q)).tolist()
                    for name, actor in self.task._actor_manager.actors.items()
                },
                "tactile": tactile_state,
            }
        )
        self.contact_trace.append(trace_entry)

        left_contact = tactile_state.get("left_tactile", {}).get("center_has_contact", False)
        right_contact = tactile_state.get("right_tactile", {}).get("center_has_contact", False)
        if left_contact and right_contact:
            if self.bilateral_contact_start_step is None:
                self.bilateral_contact_start_step = self.task.step_count
                self.capture("bilateral_contact_start")
            elif (
                not self.bilateral_contact_motion_captured
                and self.task.step_count - self.bilateral_contact_start_step >= 20 * self.task.cfg.save_frequency
            ):
                self.bilateral_contact_motion_captured = True
                self.capture("bilateral_contact_motion")
        elif (
            self.bilateral_contact_start_step is not None
            and right_contact
            and not left_contact
            and not self.left_contact_loss_captured
        ):
            self.left_contact_loss_captured = True
            self.capture("left_contact_loss")

    def _record_plan_failure(self):
        """Persist the first task failure, including a failed CuRobo query if any."""
        if self.plan_failure_recorded:
            return
        self.plan_failure_recorded = True
        self.capture("failure")
        tactile_check = {}
        for name, tactile in self.task._tactile_manager.tactiles.items():
            depth = self._to_numpy(tactile.sensor.data.output["height_map"].squeeze(0))
            center = depth[50:-50, 50:-50]
            tactile_check[name] = {
                "center_min": float(np.min(center)),
                "center_max": float(np.max(center)),
                "passes_nonconstant_contact_check": bool(np.min(center) != np.max(center)),
            }
        failure = {
            "step_count": int(self.task.step_count),
            "save_count": int(self.task.save_count),
            "keep_contact": bool(self.task.cfg.keep_contact),
            "max_save_frames": int(self.task.cfg.max_save_frames),
            "exceeded_max_save_frames": bool(self.task.save_count > self.task.cfg.max_save_frames - 1),
            "tactile_contact_check": tactile_check,
            "task_failure_context": self.task.last_plan_failure,
        }
        with open(self.output_dir / "plan_failure.json", "w", encoding="utf-8") as file:
            json.dump(failure, file, indent=2, allow_nan=False)
        log(f"Validation plan-failure diagnostics saved to {self.output_dir}")

    def _install(self):
        original_pre_move = self.task.pre_move
        original_move = self.task.move
        original_step = self.task._step

        def pre_move_wrapper(*args, **kwargs):
            if self.capture_no_contact:
                self.capture("no_contact")
            self.capture("t0")
            return original_pre_move(*args, **kwargs)

        def move_wrapper(*args, **kwargs):
            # BaseTask.move increments atom_id at entry, but may increment it
            # again for its trailing delay. Capture the impending atom before
            # calling it so telemetry remains attributed to the arm plan that
            # generated the recorded MotionGen result.
            planned_atom_id = int(self.task.atom_id) + 1
            planned_atom_tag = kwargs.get("tag", args[1] if len(args) > 1 else "move")
            step_count_before_move = int(self.task.step_count)
            result = original_move(*args, **kwargs)
            actions = args[0] if args else kwargs.get("actions", [])
            has_arm_action = any(getattr(action, "action", None) in {"move", "all"} for action in actions)
            if has_arm_action:
                # ``last_plan_diagnostics`` is recorded immediately after
                # MotionGen returns.  Persisting it here records successful
                # plans too, but does not alter the generated command stream.
                self.plan_trace.append({
                    "atom_id": planned_atom_id,
                    "atom_tag": planned_atom_tag,
                    "step_count_before_move": step_count_before_move,
                    "step_count_after_move": int(self.task.step_count),
                    "move_returned": bool(result),
                    "plan_success": bool(self.task.plan_success),
                    "diagnostics": self.task._robot_manager.planner.last_plan_diagnostics,
                })
            if not self.task.plan_success:
                self._record_plan_failure()
            if self.task.in_pre_move:
                self.pre_move_calls += 1
                # The first two moves in these scripted tasks are open then approach.
                # Capturing here leaves the gripper positioned immediately before close/contact.
                if self.pre_move_calls == 2:
                    self.capture("t1")
                    self.pre_contact_captured = True
            if self.pre_contact_captured and not self.post_contact_captured:
                if self.task.in_pre_move and self.pre_move_calls >= 3:
                    self.capture("t2")
                    self.post_contact_captured = True
                elif not self.task.in_pre_move:
                    self.capture("t2")
                    self.post_contact_captured = True
            return result

        self.task.pre_move = pre_move_wrapper
        self.task.move = move_wrapper

        def step_wrapper(*args, **kwargs):
            plan_success_before_step = self.task.plan_success
            result = original_step(*args, **kwargs)
            is_save = kwargs.get("is_save", args[0] if args else True)
            if is_save and plan_success_before_step:
                self._record_contact_trace()
            if plan_success_before_step and not self.task.plan_success:
                self._record_plan_failure()
            return result

        self.task._step = step_wrapper

    def finalize(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.output_dir / "contact_trace.json", "w", encoding="utf-8") as file:
            json.dump(self.contact_trace, file, indent=2, allow_nan=False)
        with open(self.output_dir / "plan_trace.json", "w", encoding="utf-8") as file:
            json.dump(self.plan_trace, file, indent=2, allow_nan=False)
        camera_frames = {}
        camera_shape_passes = []
        attachment_passes = []
        for label, snapshot in self.snapshots.items():
            frame = {}
            for name, camera in self.task._camera_manager.cameras.items():
                raw_rgb = snapshot["arrays"][f"camera/{name}/rgb"]
                observation_rgb = snapshot["arrays"][f"camera/{name}/observation_rgb"]
                target_resolution = camera.cfg.observation_resolution
                raw_shape_matches = list(raw_rgb.shape[:2]) == [camera.cfg.height, camera.cfg.width]
                observation_shape_matches = target_resolution is None or list(observation_rgb.shape[:2]) == [
                    target_resolution[1],
                    target_resolution[0],
                ]
                rgb_is_valid = bool(np.isfinite(observation_rgb).all() and np.std(observation_rgb) > 0)
                camera_shape_passes.extend([raw_shape_matches, observation_shape_matches, rgb_is_valid])
                frame[name] = {
                    "raw_rgb_shape": list(raw_rgb.shape),
                    "observation_rgb_shape": list(observation_rgb.shape),
                    "raw_shape_matches_config": raw_shape_matches,
                    "observation_shape_matches_config": observation_shape_matches,
                    "observation_rgb_is_finite_nonconstant": rgb_is_valid,
                }
            attachment = snapshot["state"]["camera_attachment_diagnostics"]
            for name, item in attachment.items():
                distance_error = abs(
                    item["reported_camera_to_mount_distance_m"]
                    - item["authored_camera_to_mount_distance_m"]
                )
                item["camera_to_mount_distance_error_m"] = distance_error
                item["passes_camera_to_mount_distance"] = distance_error <= 1.0e-5
                item["passes_plausible_camera_to_hand_distance"] = (
                    item["reported_camera_to_panda_hand_distance_m"] <= 0.2
                )
                attachment_passes.extend(
                    [
                        item["passes_camera_to_mount_distance"],
                        item["passes_plausible_camera_to_hand_distance"],
                    ]
                )
            frame["attachment"] = attachment
            camera_frames[label] = frame

        camera_names = list(self.task._camera_manager.cameras)
        dynamic_rgb = {}
        if len(self.snapshots) >= 2:
            first = next(iter(self.snapshots.values()))["arrays"]
            last = list(self.snapshots.values())[-1]["arrays"]
            for name in camera_names:
                before = first[f"camera/{name}/observation_rgb"].astype(np.float64)
                after = last[f"camera/{name}/observation_rgb"].astype(np.float64)
                dynamic_rgb[name] = {
                    "first_to_last_mean_absolute_difference": float(np.mean(np.abs(after - before))),
                    "is_dynamic": bool(np.any(after != before)),
                }

        camera_validation = {
            "passes": bool(camera_shape_passes and all(camera_shape_passes) and all(attachment_passes)),
            "criteria": {
                "all_shape_and_rgb_checks_pass": bool(camera_shape_passes and all(camera_shape_passes)),
                "all_attachment_checks_pass": bool(all(attachment_passes)),
                "camera_to_mount_distance_error_at_most_m": 1.0e-5,
                "camera_to_panda_hand_distance_at_most_m": 0.2,
            },
            "frames": camera_frames,
            "dynamic_rgb": dynamic_rgb,
        }
        with open(self.output_dir / "camera_validation.json", "w", encoding="utf-8") as file:
            json.dump(camera_validation, file, indent=2, allow_nan=False)
        timing_path = self.output_dir / "uipc_timing.json"
        if not timing_path.exists():
            try:
                timing_report = self.task.uipc_sim.get_sim_time_report(as_json=True)
                with open(timing_path, "w", encoding="utf-8") as file:
                    json.dump(timing_report, file, indent=2, allow_nan=False)
            except Exception as exc:
                log(f"Validation UIPC timing capture failed: {exc}")
        if not {"t1", "t2"}.issubset(self.snapshots):
            return
        comparison = {}
        for name in self.task._tactile_manager.tactiles:
            for data_type in ("rgb", "marker_rgb", "depth"):
                key = f"tactile/{name}/{data_type}"
                before = self.snapshots["t1"]["arrays"][key].astype(np.float64)
                after = self.snapshots["t2"]["arrays"][key].astype(np.float64)
                comparison[key] = {
                    "mean_absolute_difference": float(np.mean(np.abs(after - before))),
                    "max_absolute_difference": float(np.max(np.abs(after - before))),
                }
        with open(self.output_dir / "tactile_t1_t2_comparison.json", "w", encoding="utf-8") as file:
            json.dump(comparison, file, indent=2, allow_nan=False)

    def write_tactile_sanity_summary(self):
        """Write contact-state checks from already captured observation arrays.

        This is only used by ``--validation-tactile-sanity``.  It does not
        evaluate task success and it never advances the simulator.
        """
        labels = ("no_contact", "bilateral_contact", "release")
        if not set(labels).issubset(self.snapshots):
            missing = sorted(set(labels) - set(self.snapshots))
            raise RuntimeError(f"Tactile sanity capture is missing snapshots: {missing}")

        summary = {}
        for label in labels:
            contact = {}
            for name in self.task._tactile_manager.tactiles:
                depth = self.snapshots[label]["arrays"][f"tactile/{name}/depth"]
                center = depth[50:-50, 50:-50]
                contact[name] = {
                    "center_min_mm": float(np.min(center)),
                    "center_max_mm": float(np.max(center)),
                    "has_nonconstant_depth": bool(np.min(center) != np.max(center)),
                }
            summary[label] = contact
        all_no_contact = all(not value["has_nonconstant_depth"] for value in summary["no_contact"].values())
        all_bilateral_contact = all(
            value["has_nonconstant_depth"] for value in summary["bilateral_contact"].values()
        )
        all_release = all(not value["has_nonconstant_depth"] for value in summary["release"].values())
        payload = {
            "states": summary,
            "passes": bool(all_no_contact and all_bilateral_contact and all_release),
            "criteria": {
                "no_contact_is_flat": all_no_contact,
                "bilateral_contact_is_nonconstant": all_bilateral_contact,
                "release_is_flat": all_release,
            },
        }
        with open(self.output_dir / "tactile_sanity.json", "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, allow_nan=False)

def run(task: 'BaseTask', episode_num, use_seed, start_seed, max_seed, before_close=None):
    suc_num, seed = 0, 0
    suc_map = []
    
    if start_seed != -1:
        seed = start_seed
        log(f"Starting from seed {seed}.")
    elif use_seed:
        suc_map_path = task.save_root / 'suc_map.txt'
        if suc_map_path.exists():
            with open(suc_map_path, 'r') as f:
                suc_map = f.read().strip().split(' ')
            suc_num = sum([1 for s in suc_map if s == '1'])
            seed = len(suc_map)
            log(f"Use seed with {suc_num} successful episodes. Starting from seed {seed}.")

    mean_steps = 0.0
    while suc_num < episode_num and (max_seed == -1 or seed <= max_seed):
        try:
            start_t = time.perf_counter()
            task.reset(seed=seed)
            task.play_once()
            cost_t = time.perf_counter() - start_t
        except Exception as e:
            log(f"[{suc_num:<3d}] Seed {seed} failed with error: {traceback.format_exc()}")
            suc_map.append('0')
            task.clean_cache(mean_steps=mean_steps, result='error')
        else:
            if task.plan_success and task.check_success() and not task.check_early_stop():
                task.save_to_hdf5()
                log(f"[{suc_num:<3d}] Seed {seed} success in {cost_t:.2f} s.\n"
                    f"steps: {task.step_count:<5d}, save frames: {task.save_count:<5d}.\n")
                suc_num += 1
                suc_map.append('1')
                if mean_steps > 0: 
                    mean_steps = ((suc_num - 1) * mean_steps + task.step_count) / suc_num
                else:
                    mean_steps = task.step_count
                task.clean_cache(mean_steps=mean_steps, result='success')
            else:
                log(f"[{suc_num:<3d}] Seed {seed} failed in {cost_t:.2f} s.\n"
                    f"Plan {task.plan_success}, Check {task.check_success()}")
                suc_map.append('0')
                task.clean_cache(mean_steps=mean_steps, result='fail')
        
        with open(task.save_root / 'suc_map.txt', 'w') as f:
            f.write(' '.join([s for s in suc_map]))
        
        seed += 1
    
    log(f'Complete collection, success rate: {suc_num}/{seed} ({(suc_num / seed) * 100:.2f}%)')

    # Isaac Sim can terminate the process from ``simulation_app.close()``.  Emit
    # opt-in diagnostic summaries while the task is still alive, so they are not
    # lost during normal application shutdown.
    if before_close is not None:
        before_close()
    task.close()
    simulation_app.close()

def main():
    global args_cli, task_config, task_config_file, log_path
    task_file_name = args_cli.task

    episode_num = task_config.get("episode_num", -1)
    if args_cli.episode_num != -1:
        episode_num = args_cli.episode_num
    start_seed = task_config.get("start_seed", -1)
    if args_cli.start_seed != -1:
        start_seed = args_cli.start_seed
    max_seed = task_config.get("max_seed", -1)
    if args_cli.max_seed != -1:
        max_seed = args_cli.max_seed
    
    task_config.update({
        "episode_num": episode_num,
        "start_seed": start_seed,
        "max_seed": max_seed,
    })

    task_module = importlib.import_module(f"envs.{task_file_name}")
    env_cfg:'BaseTaskCfg' = task_module.TaskCfg()
    # UIPC is constructed before ``task.reset(seed=...)``.  Propagate an
    # explicit collection start seed now so its startup is deterministic too.
    if start_seed != -1:
        env_cfg = env_cfg.replace(seed=start_seed)
    env_cfg.tactile_sensor_type = task_config.get('sensor_type', 'gsmini')
    env_cfg.save_dir = Path(task_config.get("save_dir", "./data")) / task_file_name / task_config_file.stem
    env_cfg.decimation = task_config.get("decimation", env_cfg.decimation)
    env_cfg.save_frequency = task_config.get("save_frequency", env_cfg.save_frequency)
    env_cfg.video_frequency = task_config.get("video_frequency", env_cfg.video_frequency)
    env_cfg.render_frequency = task_config.get("render_frequency", env_cfg.render_frequency)
    env_cfg.obs_data_type = task_config.get("observations", {})
    env_cfg.random_texture = task_config.get("random_texture", False)

    env_cfg.scene.num_envs = 1
    
    init_start = time.perf_counter()
    task:'BaseTask' = task_module.Task(env_cfg, mode='collect')
    init_cost = time.perf_counter() - init_start
    validation_capture = None
    if args_cli.validation_dir is not None:
        validation_capture = ValidationCapture(task, args_cli.validation_dir)
        validation_capture.capture_no_contact = args_cli.validation_tactile_sanity
    
    log_path = task.save_root / f"{time.strftime(r'%Y-%m-%d_%H:%M:%S')}.log"
    log(f"Task Name: {task_file_name}")
    log(f"Config Name: {task_config_file.stem}")
    log(f"Task Config: \n{json.dumps(task_config, ensure_ascii=False, indent=4)}\n{'-' * 20}\n")
    log(f"Env Config: \n{env_cfg}\n{'-' * 20}\n")
    log(f"Init cost {init_cost:.2f} seconds, devices: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    try:
        if args_cli.validation_marker_probe:
            if validation_capture is None:
                raise ValueError("--validation-marker-probe requires --validation-dir")
            probe_seed = start_seed if start_seed != -1 else 0
            task.reset(seed=probe_seed)
            for tactile in task._tactile_manager.tactiles.values():
                tactile.sensor.marker_motion_simulator.marker_motion_simulation()
            validation_capture.capture("marker_probe")
        elif args_cli.validation_tactile_sanity:
            if validation_capture is None:
                raise ValueError("--validation-tactile-sanity requires --validation-dir")
            probe_seed = start_seed if start_seed != -1 else 0
            task.reset(seed=probe_seed)
            validation_capture.capture("bilateral_contact")
            original_keep_contact = task.cfg.keep_contact
            try:
                # Contact loss is expected during this explicit release probe.
                # Temporarily bypass the normal collect guard only for the
                # probe; restore it immediately and do not call play_once().
                task.cfg.keep_contact = False
                task.move(task.atom.open_gripper(0.8), tag="validation_release")
                task.delay(10)
                validation_capture.capture("release")
            finally:
                task.cfg.keep_contact = original_keep_contact
            validation_capture.write_tactile_sanity_summary()
        else:
            run(
                task,
                episode_num=episode_num,
                use_seed=task_config.get("use_seed", True),
                start_seed=start_seed,
                max_seed=max_seed,
                before_close=validation_capture.finalize if validation_capture is not None else None,
            )
    finally:
        if validation_capture is not None:
            validation_capture.finalize()

if __name__ == "__main__":
    main()
