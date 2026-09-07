#!/usr/bin/env python3
"""Diagnose controller fidelity against one published UniVTAC episode.

This tool does not call the task policy or change simulation configuration.  It
resets the requested task normally, then applies the saved published joint
targets through the current Isaac Lab controller for the recorded number of
physics ticks.  It is a diagnostic only: a successful replay is not a policy
success claim and it must not be used as a dataset-driven task workaround.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("task_name", help="Task module name, for example lift_can")
parser.add_argument("--task-config", default="contact", help="Task YAML name or path")
parser.add_argument("--reference", type=Path, required=True, help="Published HDF5 episode to replay")
parser.add_argument("--seed", type=int, required=True, help="Deterministic task reset seed")
parser.add_argument("--output-dir", type=Path, required=True, help="Diagnostic output directory")
parser.add_argument(
    "--max-reference-frames",
    type=int,
    default=None,
    help="Optional inclusive cap on published frames for a bounded diagnostic.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
args_cli.livestream = 0
args_cli.num_envs = 1

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import h5py
import importlib
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


def task_pose(task, actor_name: str) -> np.ndarray:
    return np.asarray(task._actor_manager.actors[actor_name].get_pose().tolist(), dtype=np.float64)


def main() -> None:
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

    task = task_module.Task(env_cfg, mode="eval_test")
    started_at = time.time()
    try:
        task.reset(seed=args_cli.seed)
        if not task.plan_success:
            raise RuntimeError("The normal pre-move failed before reference-controller replay began")

        with h5py.File(args_cli.reference, "r") as reference:
            joint = np.asarray(reference["embodiment/joint"], dtype=np.float32)
            reference_steps = np.asarray(reference["step"], dtype=np.int64)
            actor_names = sorted(reference["actor"].keys())
            actors = {name: np.asarray(reference[f"actor/{name}"], dtype=np.float32) for name in actor_names}
            if joint.ndim != 2 or joint.shape[1] < 9:
                raise ValueError(f"Expected [frames, >=9] embodiment/joint, got {joint.shape}")
            if len(reference_steps) != len(joint):
                raise ValueError("Reference step and joint arrays have different lengths")

            frame_count = len(joint)
            if args_cli.max_reference_frames is not None:
                frame_count = min(frame_count, args_cli.max_reference_frames)
            if frame_count < 2:
                raise ValueError("Need at least two reference frames")
            default_ticks = int(np.median(np.diff(reference_steps[:frame_count])))
            if default_ticks < 1:
                raise ValueError(f"Invalid reference capture cadence: {default_ticks}")

            # The normal policy writes both CuRobo position *and* velocity
            # targets.  Reconstruct the published per-joint velocity over the
            # recorded interval so this probe exercises that same controller
            # interface, rather than a position-only hold.
            reference_velocity = np.zeros_like(joint[:frame_count])
            for index in range(frame_count - 1):
                ticks = int(reference_steps[index + 1] - reference_steps[index])
                if ticks < 1:
                    raise ValueError(f"Reference step cadence is non-positive at frame {index}: {ticks}")
                reference_velocity[index] = (joint[index + 1] - joint[index]) / (ticks * task.cfg.sim.dt)

            trace = []
            for index in range(frame_count):
                ticks = (
                    int(reference_steps[index + 1] - reference_steps[index])
                    if index + 1 < frame_count
                    else default_ticks
                )
                if ticks < 1:
                    raise ValueError(f"Reference step cadence is non-positive at frame {index}: {ticks}")

                target = torch.as_tensor(joint[index], dtype=torch.float32, device=task.device)
                velocity = torch.as_tensor(reference_velocity[index], dtype=torch.float32, device=task.device)
                arm_target = target[:7]
                finger_target = target[7:9]
                arm_velocity = velocity[:7]
                finger_velocity = velocity[7:9]
                for _ in range(ticks):
                    task._robot_manager.set_arm(arm_target, arm_velocity)
                    task._robot_manager.set_gripper(finger_target, finger_velocity)
                    task._step(is_save=False)
                    if not task.plan_success:
                        raise RuntimeError(f"Task became invalid during reference frame {index}")

                # Normal collection refreshes Isaac Lab's scene data when it
                # captures each reference frame. Keep that cadence here so
                # subsequent controller commands and telemetry see current
                # simulator state. This refreshes render/observation buffers;
                # it does not modify physics, targets, or success criteria.
                task._update_render()

                actual_joint = task._robot_manager.get_qpos()[0].numpy()
                actual_ee_pose = task._robot_manager.get_ee_pose().tolist()
                actual_gripper_center_pose = task._robot_manager.get_gripper_center_pose().tolist()
                actual_actors = {name: task_pose(task, name) for name in actor_names}
                trace.append(
                    {
                        "reference_frame": index,
                        "reference_step": int(reference_steps[index]),
                        "ticks_applied": ticks,
                        "reference_joint": joint[index].tolist(),
                        "reference_joint_velocity": reference_velocity[index].tolist(),
                        "actual_joint": actual_joint.tolist(),
                        "joint_error": (actual_joint[:9] - joint[index, :9]).tolist(),
                        "actual_ee_pose_xyz_wxyz": actual_ee_pose,
                        "actual_gripper_center_pose_xyz_wxyz": actual_gripper_center_pose,
                        "reference_actor_pose_wxyz": {name: actors[name][index].tolist() for name in actor_names},
                        "actual_actor_pose_wxyz": {name: pose.tolist() for name, pose in actual_actors.items()},
                    }
                )

        joint_error = np.asarray([entry["joint_error"] for entry in trace], dtype=np.float64)
        actor_position_error = {
            name: {
                "max_mm": float(
                    max(
                        np.linalg.norm(
                            np.asarray(entry["actual_actor_pose_wxyz"][name][:3])
                            - np.asarray(entry["reference_actor_pose_wxyz"][name][:3])
                        )
                        for entry in trace
                    )
                    * 1000.0
                ),
                "final_mm": float(
                    np.linalg.norm(
                        np.asarray(trace[-1]["actual_actor_pose_wxyz"][name][:3])
                        - np.asarray(trace[-1]["reference_actor_pose_wxyz"][name][:3])
                    )
                    * 1000.0
                ),
            }
            for name in trace[0]["actual_actor_pose_wxyz"]
        }
        summary = {
            "diagnostic_only": True,
            "task_name": args_cli.task_name,
            "task_config": str(task_config_path),
            "reference": str(args_cli.reference),
            "seed": args_cli.seed,
            "physics_dt_s": float(task.cfg.sim.dt),
            "reference_frames": len(trace),
            "reference_capture_ticks": default_ticks,
            "final_task_step": int(task.step_count),
            "max_abs_joint_error": np.max(np.abs(joint_error), axis=0).tolist(),
            "final_joint_error": joint_error[-1].tolist(),
            "actor_position_error": actor_position_error,
            "elapsed_s": time.time() - started_at,
        }
        (output_dir / "reference_controller_trace.json").write_text(
            json.dumps(trace, indent=2, default=as_jsonable), encoding="utf-8"
        )
        (output_dir / "reference_controller_summary.json").write_text(
            json.dumps(summary, indent=2, default=as_jsonable), encoding="utf-8"
        )
        print(json.dumps(summary, indent=2, default=as_jsonable))
    finally:
        task.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
