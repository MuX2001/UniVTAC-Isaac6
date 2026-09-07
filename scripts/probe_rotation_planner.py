#!/usr/bin/env python3
"""Compare non-executed CuRobo plans for a task's first rotate target.

This diagnostic resets a task through its normal pre-grasp sequence, constructs
the next rotate target exactly as ``BaseTask.gripper_rotate`` does, and asks
CuRobo to plan it with three collision-world inputs.  It never sends a planned
waypoint to the robot, changes task configuration, or advances physics after
the normal reset.  Results are for root-cause diagnosis only.
"""

from __future__ import annotations

import argparse
import importlib
import json
import time
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("task_name", help="Task module name, for example lift_can")
parser.add_argument("--task-config", default="contact", help="Task YAML name or path")
parser.add_argument("--reference", type=Path, required=True, help="Published HDF5 episode for comparison")
parser.add_argument("--seed", type=int, required=True, help="Deterministic task reset seed")
parser.add_argument("--output-dir", type=Path, required=True, help="Diagnostic output directory")
parser.add_argument(
    "--reference-rotate-start-frame",
    type=int,
    default=None,
    help="Reference frame immediately before the rotate command stream (diagnostic only)",
)
parser.add_argument(
    "--reference-rotate-end-frame",
    type=int,
    default=None,
    help="Reference frame observed at the rotate endpoint (diagnostic only)",
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
from curobo.types.math import Pose as CuroboPose
from curobo.types.robot import JointState
from curobo.util.usd_helper import WorldConfig
from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig


def load_task_config(config_name: str) -> tuple[dict, Path]:
    config_path = Path(config_name)
    if not config_path.suffix:
        config_path = Path(__file__).parent.parent / "task_config" / f"{config_name}.yml"
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file), config_path


def to_jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def resample_joint_path(path: np.ndarray, count: int) -> np.ndarray:
    """Linearly resample a joint path only for cadence-normalized comparison."""
    if len(path) == count:
        return path
    source_t = np.linspace(0.0, 1.0, len(path))
    target_t = np.linspace(0.0, 1.0, count)
    return np.stack(
        [np.interp(target_t, source_t, path[:, joint]) for joint in range(path.shape[1])], axis=1
    )


def rotation_path_metrics(
    positions: np.ndarray,
    reference_joint: np.ndarray,
    start_frame: int | None,
    end_frame: int | None,
) -> dict | None:
    """Compare a generated path to the published two-tick hold command stream.

    The HDF records every two physics steps.  A held target stream is rebuilt
    from the pre-rotate sample through (but excluding) the endpoint sample;
    the endpoint itself is reported separately.  Generated and reference
    paths may have different waypoint counts, so the path comparison is
    normalized by progression rather than claimed as a tick-perfect replay.
    """
    if start_frame is None or end_frame is None:
        return None
    if start_frame < 0 or end_frame <= start_frame or end_frame >= len(reference_joint):
        raise ValueError("reference rotate frame interval is outside the HDF joint array")
    held_reference = np.repeat(reference_joint[start_frame:end_frame], 2, axis=0)
    normalized_reference = resample_joint_path(held_reference, len(positions))
    delta = positions - normalized_reference
    endpoint_delta = positions[-1] - reference_joint[end_frame]
    return {
        "reference_start_frame": start_frame,
        "reference_endpoint_frame": end_frame,
        "reference_held_command_count": int(len(held_reference)),
        "generated_command_count": int(len(positions)),
        "cadence_normalized_l2_rms": float(np.sqrt(np.mean(np.sum(delta * delta, axis=1)))),
        "cadence_normalized_l2_max": float(np.sqrt(np.sum(delta * delta, axis=1)).max()),
        "endpoint_joint_l2": float(np.linalg.norm(endpoint_delta)),
    }


def summarize_plan(
    result,
    reference_joint: np.ndarray,
    reference_rotate_start_frame: int | None = None,
    reference_rotate_end_frame: int | None = None,
) -> dict:
    """Return compact, non-executing MotionGen result evidence."""
    # Some MotionGen configuration combinations are explicitly unsupported
    # after a CUDA-graph solve.  CuRobo reports those as ``None`` rather than
    # raising a Python exception.  Keep that negative result in the report so
    # one unsupported diagnostic option cannot hide the other read-only plans.
    if result is None:
        return {
            "success": False,
            "status": "unsupported_or_no_result",
        }

    summary = {
        "success": bool(result.success.item()),
        "attempts": int(result.attempts),
        "trajopt_attempts": int(result.trajopt_attempts),
        "used_graph": bool(result.used_graph),
        "optimized_dt": result.optimized_dt,
    }
    if not summary["success"] or result.interpolated_plan is None:
        summary["status"] = str(result.status)
        return summary

    positions = result.interpolated_plan.position.detach().cpu().numpy()
    final_joint = positions[-1]
    reference_distance = np.linalg.norm(reference_joint - final_joint, axis=1)
    closest_frame = int(np.argmin(reference_distance))
    summary.update(
        {
            "interpolated_waypoints": int(positions.shape[0]),
            "start_joint": positions[0],
            "final_joint": final_joint,
            "max_joint_step": (
                np.abs(np.diff(positions, axis=0)).max(axis=0)
                if len(positions) > 1
                else np.zeros(7)
            ),
            "closest_reference_frame": closest_frame,
            "closest_reference_joint_l2": float(reference_distance[closest_frame]),
            "closest_reference_joint": reference_joint[closest_frame],
            "reference_rotate_path_metrics": rotation_path_metrics(
                positions,
                reference_joint,
                reference_rotate_start_frame,
                reference_rotate_end_frame,
            ),
        }
    )
    return summary


def pose_delta(first: np.ndarray, second: np.ndarray) -> dict:
    """Compare two xyz+wxyz poses without assigning an execution meaning."""
    first_quaternion = first[3:] / np.linalg.norm(first[3:])
    second_quaternion = second[3:] / np.linalg.norm(second[3:])
    cosine = float(np.clip(abs(np.dot(first_quaternion, second_quaternion)), -1.0, 1.0))
    return {
        "position_m": float(np.linalg.norm(first[:3] - second[:3])),
        "orientation_rad": float(2.0 * np.arccos(cosine)),
    }


def motiongen_fk(planner, joint_position: np.ndarray) -> np.ndarray:
    """Return pinned MotionGen's own end-effector FK for an arm joint vector."""
    state = JointState(
        position=torch.as_tensor(joint_position, device=planner.motion_gen.tensor_args.device).reshape(1, -1),
        joint_names=planner.active_joints_name,
    )
    kinematic_state = planner.motion_gen.compute_kinematics(state)
    return np.concatenate(
        (
            kinematic_state.ee_pos_seq[0].detach().cpu().numpy(),
            kinematic_state.ee_quat_seq[0].detach().cpu().numpy(),
        )
    )


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
            raise RuntimeError("Normal pre-move failed before planner probe")

        # Both rotation tasks expose their manipulated rigid object under one
        # of these names.  This probe remains execution-free for either task.
        actor = getattr(task, "can", None) or getattr(task, "bottle", None)
        if actor is None:
            raise ValueError(f"{args_cli.task_name} does not expose a can or bottle rotation actor")
        robot = task._robot_manager
        planner = robot.planner
        rpy = [0.0, (70.0 / 180.0 * np.pi) / 4.0, 0.0]
        gripper_center = robot.get_gripper_center_pose()
        target_center = gripper_center.add_rotation(rpy, coord=actor.get_pose())
        target_center.q = gripper_center.q.copy()
        target_pose = robot.gripper_center_to_ee(target_center)

        with h5py.File(args_cli.reference, "r") as reference:
            reference_joint = np.asarray(reference["embodiment/joint"], dtype=np.float32)[:, :7]
            # This is the source task's ``Pose`` field: xyz + wxyz.
            reference_ee_wxyz = np.asarray(reference["embodiment/ee"], dtype=np.float32)

        current_joint = robot.get_qpos()[0, :7].numpy()
        original_get_world = planner.get_curr_world_cfg

        def ground_only_world():
            return planner.usd_helper.get_obstacles_from_stage(
                only_paths=["/World/envs/env_0/ground_plate"],
                reference_prim_path=planner.robot_prime_path,
            ).get_collision_check_world()

        variants = {
            "ground_and_all_actors": original_get_world,
            "ground_only_diagnostic": ground_only_world,
            "empty_world_diagnostic": WorldConfig,
        }
        results = {}
        for name, get_world in variants.items():
            planner.get_curr_world_cfg = get_world
            plan = robot.plan_arm(target_pose, time_dilation_factor=0.5)
            result = {
                "status": plan["status"],
                "diagnostics": plan["diagnostics"],
            }
            if plan["status"] == "Success":
                positions = plan["position"].detach().cpu().numpy()
                final_joint = positions[-1]
                reference_distance = np.linalg.norm(reference_joint - final_joint, axis=1)
                closest_frame = int(np.argmin(reference_distance))
                result.update(
                    {
                        "interpolated_waypoints": int(positions.shape[0]),
                        "start_joint": positions[0],
                        "final_joint": final_joint,
                        "max_joint_step": (
                            np.abs(np.diff(positions, axis=0)).max(axis=0)
                            if len(positions) > 1
                            else np.zeros(7)
                        ),
                        "closest_reference_frame": closest_frame,
                        "closest_reference_joint_l2": float(reference_distance[closest_frame]),
                        "closest_reference_joint": reference_joint[closest_frame],
                        "reference_rotate_path_metrics": rotation_path_metrics(
                            positions,
                            reference_joint,
                            args_cli.reference_rotate_start_frame,
                            args_cli.reference_rotate_end_frame,
                        ),
                    }
                )
            results[name] = result

        planner.get_curr_world_cfg = original_get_world
        planner.update_world()
        active_joint_ids = np.array([planner.all_joints.index(name) for name in planner.active_joints_name])
        current_position = robot.robot.data.joint_pos[0, : robot.robot.num_joints - 2][active_joint_ids].reshape(1, -1)
        current_velocity = robot.robot.data.joint_vel[0, : robot.robot.num_joints - 2][active_joint_ids].reshape(1, -1)
        start_state = JointState(
            position=current_position,
            velocity=current_velocity,
            acceleration=torch.zeros_like(current_position),
            jerk=torch.zeros_like(current_position),
            joint_names=planner.active_joints_name,
        )
        planner_target = results["ground_and_all_actors"]["diagnostics"]["target_pose_robot_base_xyz_wxyz"]
        goal = CuroboPose.from_list(planner_target)
        current_fk = motiongen_fk(planner, current_joint)
        fk_comparison = {
            "convention": {
                "motiongen": "xyz_wxyz",
                "task_target": "xyz_wxyz",
                "reference_observation": "xyz_wxyz",
            },
            "simulator_current_ee_xyz_wxyz": robot.get_ee_pose().tolist(),
            "simulator_current_gripper_center_xyz_wxyz": robot.get_gripper_center_pose().tolist(),
            "motiongen_current_joint_fk_xyz_wxyz": current_fk,
            "motiongen_current_joint_fk_vs_target": pose_delta(
                current_fk, np.asarray(planner_target)
            ),
            "reference_frames": {},
        }
        for frame in (0, 22, 23, 53):
            observed = reference_ee_wxyz[frame]
            reference_fk = motiongen_fk(planner, reference_joint[frame])
            fk_comparison["reference_frames"][str(frame)] = {
                "reference_joint": reference_joint[frame],
                "reference_observation_ee_xyz_wxyz": observed,
                "motiongen_fk_xyz_wxyz": reference_fk,
                "motiongen_fk_vs_reference_observation": pose_delta(reference_fk, observed),
                "motiongen_fk_vs_current_rotate_target": pose_delta(reference_fk, np.asarray(planner_target)),
            }
        option_variants = {
            "default": {},
            "retract_config_instead_of_start": {"use_start_state_as_retract": False},
            "single_seed_finetune": {"parallel_finetune": False},
            "no_finetune_diagnostic": {"enable_finetune_trajopt": False},
            "graph_enabled": {"enable_graph": True},
        }
        option_results = {}
        for name, options in option_variants.items():
            plan_config = MotionGenPlanConfig(max_attempts=10, time_dilation_factor=0.5, **options)
            try:
                result = planner.motion_gen.plan_single(start_state, goal, plan_config)
                summary = summarize_plan(
                    result,
                    reference_joint,
                    args_cli.reference_rotate_start_frame,
                    args_cli.reference_rotate_end_frame,
                )
            except Exception as error:  # record an unsupported solver mode; never execute a fallback path
                summary = {
                    "success": False,
                    "status": "exception",
                    "exception_type": type(error).__name__,
                    "exception": str(error),
                }
            option_results[name] = {"plan_config_overrides": options, **summary}

        report = {
            "diagnostic_only": True,
            "task_name": args_cli.task_name,
            "task_config": str(task_config_path),
            "reference": str(args_cli.reference),
            "seed": args_cli.seed,
            "physics_dt_s": float(task.cfg.sim.dt),
            "step_count_after_normal_pre_move": int(task.step_count),
            "current_joint_after_normal_pre_move": current_joint,
            "first_rotate_target_ee_xyz_wxyz": target_pose.tolist(),
            "reference_rotate_frame_interval": {
                "start": args_cli.reference_rotate_start_frame,
                "endpoint": args_cli.reference_rotate_end_frame,
            },
            "fk_comparison": fk_comparison,
            "variants": results,
            "motiongen_option_variants": option_results,
            "elapsed_s": time.time() - started_at,
        }
        (output_dir / "rotation_planner_probe.json").write_text(
            json.dumps(report, indent=2, default=to_jsonable), encoding="utf-8"
        )
        print(json.dumps(report, indent=2, default=to_jsonable))
    finally:
        task.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
