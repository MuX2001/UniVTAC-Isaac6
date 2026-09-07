#!/usr/bin/env python3
"""Capture normal CuRobo rotation commands and compare them with a public HDF.

The probe resets the requested task normally, performs its normal close action,
then executes only the requested number of original rotation segments.  It
records the CuRobo position/velocity commands before the existing task driver
executes them.  It does not replace commands, replay HDF commands, or change
physics, task conditions, RNG, planner settings, or success logic.
"""

from __future__ import annotations

import argparse
import importlib
import json
import time
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("task_name", help="Task module name, for example lift_bottle")
parser.add_argument("--task-config", default="contact", help="Task YAML name or path")
parser.add_argument("--reference", type=Path, required=True, help="Published HDF5 episode")
parser.add_argument("--seed", type=int, required=True, help="Deterministic task reset seed")
parser.add_argument("--output-dir", type=Path, required=True, help="Diagnostic output directory")
parser.add_argument("--rotations", type=int, default=2, help="Number of original rotation segments to execute")
parser.add_argument(
    "--trace-final-ticks",
    type=int,
    default=0,
    help="Read-only number of final PhysX/controller ticks retained per rotation (default: disabled)",
)
parser.add_argument(
    "--probe-second-start-variants",
    action="store_true",
    help="Execute only rotation one, then plan but do not execute second-query start-state variants",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
args_cli.livestream = 0
args_cli.num_envs = 1

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import h5py
import numpy as np
import torch
import yaml


def load_task_config(config_name: str) -> tuple[dict, Path]:
    config_path = Path(config_name)
    if not config_path.suffix:
        config_path = Path(__file__).parent.parent / "task_config" / f"{config_name}.yml"
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file), config_path


def as_jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def to_numpy(value):
    """Convert live Isaac/PhysX diagnostic buffers without changing them."""
    if hasattr(value, "torch"):
        value = value.torch
    elif value.__class__.__module__.startswith("warp"):
        import warp as wp

        value = wp.to_torch(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def decode_tag(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def reference_rotate_segments(reference: h5py.File) -> list[dict]:
    atom_ids = np.asarray(reference["atom/id"], dtype=np.int64)
    tags = [decode_tag(value) for value in np.asarray(reference["atom/tag"])]
    steps = np.asarray(reference["step"], dtype=np.int64)
    joints = np.asarray(reference["embodiment/joint"], dtype=np.float32)[:, :7]
    segments = []
    start = 0
    while start < len(tags):
        end = start + 1
        while end < len(tags) and atom_ids[end] == atom_ids[start] and tags[end] == tags[start]:
            end += 1
        if tags[start] == "rotate":
            previous = max(0, start - 1)
            held_position = []
            held_velocity = []
            for index in range(previous, end - 1):
                ticks = int(steps[index + 1] - steps[index])
                if ticks < 1:
                    raise ValueError(f"Non-positive HDF step cadence at frame {index}")
                velocity = (joints[index + 1] - joints[index]) / (ticks / 120.0)
                held_position.extend([joints[index]] * ticks)
                held_velocity.extend([velocity] * ticks)
            segments.append(
                {
                    "atom_id": int(atom_ids[start]),
                    "start_frame": int(previous),
                    "first_rotate_frame": int(start),
                    "endpoint_frame": int(end - 1),
                    "start_step": int(steps[previous]),
                    "endpoint_step": int(steps[end - 1]),
                    "held_position": np.asarray(held_position, dtype=np.float32),
                    "held_velocity": np.asarray(held_velocity, dtype=np.float32),
                    "endpoint_joint": joints[end - 1],
                }
            )
        start = end
    return segments


def resample(path: np.ndarray, count: int) -> np.ndarray:
    if len(path) == count:
        return path
    source_t = np.linspace(0.0, 1.0, len(path))
    target_t = np.linspace(0.0, 1.0, count)
    return np.stack(
        [np.interp(target_t, source_t, path[:, joint]) for joint in range(path.shape[1])], axis=1
    )


def path_metrics(generated: np.ndarray, reference: np.ndarray) -> dict:
    normalized_reference = resample(reference, len(generated))
    norm = np.linalg.norm(generated - normalized_reference, axis=1)
    return {
        "generated_count": int(len(generated)),
        "reference_held_count": int(len(reference)),
        "cadence_normalized_joint_l2_rms": float(np.sqrt(np.mean(norm * norm))),
        "cadence_normalized_joint_l2_max": float(norm.max()),
        "per_joint_max_abs": np.max(np.abs(generated - normalized_reference), axis=0),
    }


def state_snapshot(task, actor) -> dict:
    return {
        "task_step": int(task.step_count),
        "joint_position": task._robot_manager.get_qpos()[0].numpy(),
        "gripper_center_pose_xyz_wxyz": task._robot_manager.get_gripper_center_pose().tolist(),
        "actor_pose_xyz_wxyz": actor.get_pose().tolist(),
    }


def main() -> None:
    if args_cli.rotations < 1 or args_cli.rotations > 4:
        raise ValueError("--rotations must be between 1 and the task's original 4 segments")
    if args_cli.trace_final_ticks < 0:
        raise ValueError("--trace-final-ticks must be non-negative")
    task_config, task_config_path = load_task_config(args_cli.task_config)
    task_module = importlib.import_module(f"envs.{args_cli.task_name}")
    output_dir = args_cli.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = task_module.TaskCfg()
    env_cfg.save_dir = output_dir / "task_runtime"
    env_cfg.decimation = task_config.get("decimation", env_cfg.decimation)
    env_cfg.obs_data_type = task_config.get("observations", {})
    env_cfg.save_frequency = task_config.get("save_frequency", env_cfg.save_frequency)
    env_cfg.video_frequency = 0
    env_cfg.render_frequency = 0
    env_cfg.random_texture = task_config.get("random_texture", False)
    env_cfg.scene.num_envs = 1
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    with h5py.File(args_cli.reference, "r") as reference:
        reference_segments = reference_rotate_segments(reference)
    if len(reference_segments) < args_cli.rotations:
        raise ValueError(
            f"Reference contains {len(reference_segments)} rotate segments, requested {args_cli.rotations}"
        )

    task = task_module.Task(env_cfg, mode="eval_test")
    started_at = time.time()
    captured = []
    try:
        task.reset(seed=args_cli.seed)
        if not task.plan_success:
            raise RuntimeError("Normal pre-move failed before command capture")
        actor = getattr(task, "bottle", None) or getattr(task, "can", None)
        if actor is None:
            raise ValueError(f"{args_cli.task_name} does not expose a bottle or can actor")

        task.move(task.atom.close_gripper(), is_save=False)
        if not task.plan_success:
            raise RuntimeError("Normal close action failed before rotation capture")

        robot = task._robot_manager
        original_plan_arm = robot.plan_arm
        original_set_arm = robot.set_arm
        original_step = task._step
        current_arm_command = None
        command_indices = {}
        final_tick_trace = {}

        def capture_plan(*args, **kwargs):
            if captured and "post_execution" not in captured[-1]:
                captured[-1]["post_execution"] = state_snapshot(task, actor)
            before = state_snapshot(task, actor)
            plan = original_plan_arm(*args, **kwargs)
            if task.atom_tag == "rotate":
                entry = {
                    "atom_id": int(task.atom_id),
                    "atom_tag": task.atom_tag,
                    "pre_execution": before,
                    "target_pose_xyz_wxyz": args[0].tolist(),
                    "status": plan["status"],
                    "diagnostics": plan.get("diagnostics"),
                }
                if plan["status"] == "Success":
                    entry["position_commands"] = plan["position"].detach().cpu().numpy()
                    entry["velocity_commands"] = plan["velocity"].detach().cpu().numpy()
                captured.append(entry)
            return plan

        def trace_set_arm(pos, vel=None, *args, **kwargs):
            nonlocal current_arm_command
            result = original_set_arm(pos, vel, *args, **kwargs)
            if args_cli.trace_final_ticks and task.atom_tag == "rotate":
                atom_id = int(task.atom_id)
                command_index = command_indices.get(atom_id, 0)
                command_indices[atom_id] = command_index + 1
                current_arm_command = {
                    "atom_id": atom_id,
                    "command_index": command_index,
                    "position": to_numpy(pos).reshape(-1),
                    "velocity": None if vel is None else to_numpy(vel).reshape(-1),
                }
            return result

        def trace_step(*args, **kwargs):
            result = original_step(*args, **kwargs)
            if not args_cli.trace_final_ticks or current_arm_command is None or task.atom_tag != "rotate":
                return result
            atom_id = int(task.atom_id)
            try:
                live_position = to_numpy(robot.robot.root_physx_view.get_dof_positions())[0]
                live_velocity = to_numpy(robot.robot.root_physx_view.get_dof_velocities())[0]
                arm_ids = to_numpy(robot._arm_ids).astype(np.int64)
                actual_position = live_position[arm_ids]
                actual_velocity = live_velocity[arm_ids]
                position_target = current_arm_command["position"]
                velocity_target = current_arm_command["velocity"]
                effort = {}
                for name, actuator in robot.robot.actuators.items():
                    values = {}
                    for attribute in ("computed_effort", "applied_effort"):
                        value = getattr(actuator, attribute, None)
                        if value is not None:
                            values[attribute] = to_numpy(value).tolist()
                    if values:
                        effort[name] = values
                entry = {
                    "task_step": int(task.step_count),
                    "command_index": current_arm_command["command_index"],
                    "position_target": position_target.tolist(),
                    "velocity_target": None if velocity_target is None else velocity_target.tolist(),
                    "actual_position": actual_position.tolist(),
                    "actual_velocity": actual_velocity.tolist(),
                    "position_target_error": (position_target - actual_position).tolist(),
                    "position_target_error_l2": float(np.linalg.norm(position_target - actual_position)),
                    "actual_velocity_l2": float(np.linalg.norm(actual_velocity)),
                    "actuator_effort": effort,
                }
            except Exception as error:
                entry = {
                    "task_step": int(task.step_count),
                    "command_index": current_arm_command["command_index"],
                    "read_error": f"{type(error).__name__}: {error}",
                }
            trace = final_tick_trace.setdefault(str(atom_id), [])
            trace.append(entry)
            if len(trace) > args_cli.trace_final_ticks:
                del trace[:-args_cli.trace_final_ticks]
            return result

        robot.plan_arm = capture_plan
        robot.set_arm = trace_set_arm
        task._step = trace_step
        original_total_theta = 70.0 / 180.0 * np.pi
        if args_cli.probe_second_start_variants:
            task.gripper_rotate(actor, original_total_theta / 4.0, steps=1, is_save=False)
            if captured and "post_execution" not in captured[-1]:
                captured[-1]["post_execution"] = state_snapshot(task, actor)
            if not task.plan_success or len(captured) != 1:
                raise RuntimeError("Normal first rotation did not complete before second-query probe")

            gripper_center = robot.get_gripper_center_pose()
            second_target_center = gripper_center.add_rotation(
                [0.0, original_total_theta / 4.0, 0.0], coord=actor.get_pose()
            )
            second_target_center.q = gripper_center.q.copy()
            second_target = robot.gripper_center_to_ee(second_target_center)
            live_position = robot.robot.data.joint_pos[0, : robot.robot.num_joints - 2].detach()
            live_velocity = robot.robot.data.joint_vel[0, : robot.robot.num_joints - 2].detach()
            planned_endpoint = torch.as_tensor(
                captured[0]["position_commands"][-1], device=live_position.device
            )
            planned_velocity = torch.as_tensor(
                captured[0]["velocity_commands"][-1], device=live_velocity.device
            )
            start_variants = {
                "live_position_live_velocity": (live_position, live_velocity),
                "live_position_zero_velocity": (live_position, torch.zeros_like(live_velocity)),
                "previous_plan_endpoint_zero_velocity": (
                    planned_endpoint,
                    torch.zeros_like(planned_velocity),
                ),
                "previous_plan_endpoint_planned_velocity": (planned_endpoint, planned_velocity),
            }
            second_reference = reference_segments[1]
            variant_results = {}
            planner = robot.planner
            for name, (start_position, start_velocity) in start_variants.items():
                planner.motion_gen.reset_seed()
                result = planner.plan_path(
                    curr_joint_pos=start_position,
                    curr_joint_vel=start_velocity,
                    target_ee_pose=second_target,
                    real_robot_pose=robot.root_pose,
                    time_dilation_factor=0.5,
                )
                entry = {
                    "success": bool(result.success.item()),
                    "status": None if result.status is None else str(result.status),
                    "start_position": to_numpy(start_position),
                    "start_velocity": to_numpy(start_velocity),
                    "diagnostics": planner.last_plan_diagnostics,
                }
                if entry["success"] and result.interpolated_plan is not None:
                    position = result.interpolated_plan.position.detach().cpu().numpy()
                    velocity = result.interpolated_plan.velocity.detach().cpu().numpy()
                    entry["position"] = {
                        **path_metrics(position, second_reference["held_position"]),
                        "endpoint_joint_l2": float(
                            np.linalg.norm(position[-1] - second_reference["endpoint_joint"])
                        ),
                    }
                    entry["velocity"] = path_metrics(velocity, second_reference["held_velocity"])
                variant_results[name] = entry

            report = {
                "diagnostic_only": True,
                "normal_first_rotation_executed_unchanged": True,
                "second_rotation_candidate_commands_executed": False,
                "reference_commands_executed": False,
                "seed_reset_before_each_candidate": True,
                "task_name": args_cli.task_name,
                "task_config": str(task_config_path),
                "reference": str(args_cli.reference),
                "seed": args_cli.seed,
                "physics_dt_s": float(task.cfg.sim.dt),
                "second_target_pose_xyz_wxyz": second_target.tolist(),
                "normal_first_rotation": captured[0],
                "second_start_variants": variant_results,
                "elapsed_s": time.time() - started_at,
            }
            (output_dir / "second_rotation_start_variants.json").write_text(
                json.dumps(report, indent=2, default=as_jsonable), encoding="utf-8"
            )
            print(
                json.dumps(
                    {key: value for key, value in report.items() if key != "normal_first_rotation"},
                    indent=2,
                    default=as_jsonable,
                )
            )
            return

        task.gripper_rotate(
            actor,
            original_total_theta * args_cli.rotations / 4.0,
            steps=args_cli.rotations,
            is_save=False,
        )
        if captured and "post_execution" not in captured[-1]:
            captured[-1]["post_execution"] = state_snapshot(task, actor)
        if not task.plan_success:
            raise RuntimeError("A normal rotation plan failed during command capture")
        if len(captured) != args_cli.rotations:
            raise RuntimeError(f"Expected {args_cli.rotations} captured rotations, got {len(captured)}")

        comparisons = []
        for local, reference in zip(captured, reference_segments[: args_cli.rotations], strict=True):
            position = np.asarray(local["position_commands"], dtype=np.float32)
            velocity = np.asarray(local["velocity_commands"], dtype=np.float32)
            comparisons.append(
                {
                    "local_atom_id": local["atom_id"],
                    "reference_atom_id": reference["atom_id"],
                    "reference_frames": [reference["start_frame"], reference["endpoint_frame"]],
                    "reference_steps": [reference["start_step"], reference["endpoint_step"]],
                    "position": {
                        **path_metrics(position, reference["held_position"]),
                        "endpoint_joint_l2": float(
                            np.linalg.norm(position[-1] - reference["endpoint_joint"])
                        ),
                    },
                    "velocity": path_metrics(velocity, reference["held_velocity"]),
                    "local_actor_start_xyz": local["pre_execution"]["actor_pose_xyz_wxyz"][:3],
                    "local_actor_end_xyz": local["post_execution"]["actor_pose_xyz_wxyz"][:3],
                }
            )

        report = {
            "diagnostic_only": True,
            "normal_commands_executed_unchanged": True,
            "reference_commands_executed": False,
            "task_name": args_cli.task_name,
            "task_config": str(task_config_path),
            "reference": str(args_cli.reference),
            "seed": args_cli.seed,
            "physics_dt_s": float(task.cfg.sim.dt),
            "rotation_segments": args_cli.rotations,
            "trace_final_ticks": args_cli.trace_final_ticks,
            "final_tick_trace": final_tick_trace,
            "comparisons": comparisons,
            "captured_plans": captured,
            "elapsed_s": time.time() - started_at,
        }
        (output_dir / "rotation_command_streams.json").write_text(
            json.dumps(report, indent=2, default=as_jsonable), encoding="utf-8"
        )
        print(
            json.dumps(
                {key: value for key, value in report.items() if key != "captured_plans"},
                indent=2,
                default=as_jsonable,
            )
        )
    finally:
        task.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
