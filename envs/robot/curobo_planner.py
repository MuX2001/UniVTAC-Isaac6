from curobo.geom.transform import pose_multiply
import numpy as np
import transforms3d as t3d
from curobo.types.robot import JointState
from curobo.util.usd_helper import UsdHelper, WorldConfig
from curobo.types.math import Pose as CuroboPose
from curobo.geom.types import Mesh
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
    PoseCostMetric,
)
import torch
import yaml
from curobo.util import logger
from copy import deepcopy

from pydantic import constr
logger.setup_logger(level="error", logger_name="curobo")

from pathlib import Path
from ..utils.transforms import *
from isaaclab.utils import configclass

@configclass
class CuroboPlannerCfg:
    dt: float = 1/120
    yaml_path: str = None
    robot_prime_path: str = "/World/robot"

    all_joints_name: list[str] = None
    active_joints_name: list[str] = None

    time_dilation_factor: float = 1.0

class CuroboPlanner:
    def __init__(
        self,
        task: 'BaseTask',
        cfg: CuroboPlannerCfg,
        robot_origin_pose:Pose,
    ):
        super().__init__()
        logger.setup_logger(level="error", logger_name="'curobo")

        self.cfg = cfg
        self.task = task
        self.dt = cfg.dt
        self.robot_prime_path = cfg.robot_prime_path
        self.robot_origin_pose = robot_origin_pose
        self.active_joints_name = cfg.active_joints_name
        self.all_joints = cfg.all_joints_name
        # translate from baselink to arm's base
        with open(self.cfg.yaml_path, "r") as f:
            yml_data = yaml.safe_load(f)

        file_dir = Path(self.cfg.yaml_path).parent
        urdf_path = yml_data['robot_cfg']['kinematics']['urdf_path']
        if not Path(urdf_path).is_absolute():
            yml_data['robot_cfg']['kinematics']['urdf_path'] = str(file_dir / urdf_path)
        collision_spheres = yml_data['robot_cfg']['kinematics']['collision_spheres']
        if not Path(collision_spheres).is_absolute():
            yml_data['robot_cfg']['kinematics']['collision_spheres'] = str(file_dir / collision_spheres)

        self.frame_bias = yml_data["planner"]["frame_bias"]

        self.usd_helper = UsdHelper()
        self.usd_helper.load_stage(self.task.scene.stage)

        motion_gen_config = MotionGenConfig.load_from_robot_config(
            robot_cfg=yml_data,
            world_model=self.get_curr_world_cfg(),
            interpolation_dt=self.dt,
            position_threshold=0.001,
            rotation_threshold=0.01,
            high_precision=True,
            collision_checker_type=CollisionCheckerType.MESH,
            # Match the upstream UniVTAC planning configuration.  A 0.4 m
            # activation shell is far larger than this robot/object scene and
            # changed the generated trajectory even when no collision occurred.
            collision_activation_distance=0.0,
        )
        self.motion_gen = MotionGen(motion_gen_config)
        self.motion_gen.warmup()
        # Populated for every query so callers can record why a failed plan was
        # rejected.  This is telemetry only; it does not alter a MotionGen
        # configuration or retry policy.
        self.last_plan_diagnostics = None
        self.last_world_diagnostics = None
    
    def reset(self):
        self.motion_gen.reset()

    def get_curr_world_cfg(self):
        """Build the collision world used by CuRobo from current scene state.

        The original UniVTAC planner includes the ground plate and actor meshes.
        The Isaac Sim 6 port temporarily replaced that world with a cuboid at
        x=-1000, which silently disabled environment collision checks and made
        its trajectories incomparable to the official dataset.  Retain actor
        meshes even if Isaac Sim 6 cannot convert the authored ground plate.
        """
        diagnostics = {"ground_plate": "unavailable", "actor_meshes": [], "errors": []}
        try:
            obstacles = self.usd_helper.get_obstacles_from_stage(
                only_paths=["/World/envs/env_0/ground_plate"],
                reference_prim_path=self.robot_prime_path,
            ).get_collision_check_world()
            diagnostics["ground_plate"] = "stage_mesh"
        except Exception as exc:
            obstacles = WorldConfig()
            diagnostics["errors"].append(f"ground_plate: {type(exc).__name__}: {exc}")

        for name, actor in self.task._actor_manager.actors.items():
            try:
                vertices = np.asarray(actor.vertices)
                if vertices.size == 0:
                    raise ValueError("actor has no surface vertices")
                mesh = Mesh.from_pointcloud(vertices.reshape(-1, 3), pitch=0.005, name=name)
                obstacles.add_obstacle(mesh)
                diagnostics["actor_meshes"].append(name)
            except Exception as exc:
                diagnostics["errors"].append(f"actor {name}: {type(exc).__name__}: {exc}")
        self.last_world_diagnostics = diagnostics
        return obstacles
 
    def update_world(self):
        self.motion_gen.update_world(self.get_curr_world_cfg())

    @staticmethod
    def _diagnostic_value(value):
        """Convert a CuRobo value into a bounded JSON-safe diagnostic value."""
        if value is None or isinstance(value, (bool, int, str)):
            return value
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float):
            return value if np.isfinite(value) else repr(value)
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        if isinstance(value, np.ndarray):
            if value.size <= 32:
                return CuroboPlanner._diagnostic_value(value.tolist())
            finite = np.isfinite(value)
            finite_values = value[finite]
            return {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "finite_count": int(finite.sum()),
                "min": None if not finite_values.size else float(finite_values.min()),
                "max": None if not finite_values.size else float(finite_values.max()),
            }
        if isinstance(value, (list, tuple)):
            if len(value) > 32:
                return {"length": len(value), "preview": [CuroboPlanner._diagnostic_value(v) for v in value[:8]]}
            return [CuroboPlanner._diagnostic_value(v) for v in value]
        if isinstance(value, dict):
            return {
                str(key): CuroboPlanner._diagnostic_value(item)
                for key, item in list(value.items())[:32]
            }
        value_repr = repr(value)
        return value_repr[:1000] + ("..." if len(value_repr) > 1000 else "")

    def _record_plan_diagnostics(self, result, target_pose, joint_pos, joint_vel, plan_config):
        """Store fields exposed by MotionGenResult without interpreting success."""
        diagnostics = {
            "target_pose_robot_base_xyz_wxyz": self._diagnostic_value(target_pose.tolist()),
            "start_joint_position": self._diagnostic_value(joint_pos),
            "start_joint_velocity": self._diagnostic_value(joint_vel),
            "collision_world": self._diagnostic_value(self.last_world_diagnostics),
            "plan_config": {
                "max_attempts": int(plan_config.max_attempts),
                "time_dilation_factor": self._diagnostic_value(plan_config.time_dilation_factor),
                "has_pose_cost_metric": bool(plan_config.pose_cost_metric is not None),
            },
        }
        for field in (
            "success",
            "valid_query",
            "status",
            "attempts",
            "trajopt_attempts",
            "used_graph",
            "solve_time",
            "ik_time",
            "graph_time",
            "trajopt_time",
            "finetune_time",
            "total_time",
            "position_error",
            "rotation_error",
            "cspace_error",
            "optimized_dt",
        ):
            if hasattr(result, field):
                diagnostics[field] = self._diagnostic_value(getattr(result, field))
        for field in ("interpolated_plan", "optimized_plan", "path_buffer_last_tstep"):
            if hasattr(result, field):
                value = getattr(result, field)
                diagnostics[f"has_{field}"] = value is not None
        # CuRobo versions can expose a very different number of execution
        # waypoints in the optimized and interpolated trajectories.  Record
        # both read-only summaries so a compatibility diagnosis does not infer
        # the executed cadence from one representation alone.
        for plan_name in ("interpolated", "optimized"):
            plan = getattr(result, f"{plan_name}_plan", None)
            if plan is None:
                continue
            position = getattr(plan, "position", None)
            velocity = getattr(plan, "velocity", None)
            if position is not None:
                diagnostics[f"{plan_name}_plan_num_steps"] = int(position.shape[0])
                if position.shape[0] > 1:
                    diagnostics[f"{plan_name}_plan_max_position_step_delta"] = self._diagnostic_value(
                        torch.abs(position[1:] - position[:-1]).amax(dim=0)
                    )
            if velocity is not None:
                diagnostics[f"{plan_name}_plan_max_abs_velocity"] = self._diagnostic_value(
                    torch.abs(velocity).amax(dim=0)
                )
        if hasattr(result, "interpolation_dt"):
            diagnostics["interpolation_dt"] = self._diagnostic_value(result.interpolation_dt)
        if hasattr(result, "debug_info"):
            debug_info = result.debug_info
            diagnostics["debug_info_type"] = type(debug_info).__name__
            diagnostics["debug_info"] = self._diagnostic_value(debug_info)
        self.last_plan_diagnostics = diagnostics

    def plan_path(
        self,
        curr_joint_pos: torch.Tensor,
        curr_joint_vel: torch.Tensor,
        target_ee_pose,
        real_robot_pose,
        pre_dis=None,
        constraint_pose=None,
        time_dilation_factor=None
    ):
        self.update_world()
        target_pose = calculate_target_pose(
            real_robot_pose, self.robot_origin_pose, target_ee_pose)
        # transformation from world to arm's base
        target_pose = target_pose.rebase(to_coord=self.robot_origin_pose).add_bias(
            self.frame_bias, coord='world', clone=False
        )
        goal_pose_of_ee = CuroboPose.from_list(target_pose.tolist())
        joint_indices = np.array([
            self.all_joints.index(name) for name in self.active_joints_name if name in self.all_joints])
        joint_pos = curr_joint_pos[joint_indices].reshape(1, -1)
        joint_vel = curr_joint_vel[joint_indices].reshape(1, -1)
        
        start_joint_states = JointState(
            position=joint_pos,
            velocity=joint_vel,
            acceleration=torch.zeros_like(joint_pos),
            jerk=torch.zeros_like(joint_pos),
            joint_names=self.active_joints_name,
        )
        # plan
        if time_dilation_factor is None:
            time_dilation_factor = self.cfg.time_dilation_factor
        plan_config = MotionGenPlanConfig(max_attempts=10, time_dilation_factor=time_dilation_factor)

        pose_cost_metric = None
        if constraint_pose is not None:
            if pre_dis is not None:
                pose_cost_metric = PoseCostMetric(
                    hold_partial_pose=True,
                    hold_vec_weight=self.motion_gen.tensor_args.to_device(constraint_pose),
                    offset_position=self.motion_gen.tensor_args.to_device([0.0, 0.0, pre_dis])
                )
            else:
                pose_cost_metric = PoseCostMetric(
                    hold_partial_pose=True,
                    hold_vec_weight=self.motion_gen.tensor_args.to_device(constraint_pose)
                )
        elif pre_dis is not None and pre_dis != 0.0:
            pose_cost_metric = PoseCostMetric.create_grasp_approach_metric(
                offset_position=pre_dis, tstep_fraction=0.6, linear_axis=2)

        if pose_cost_metric is not None:
            plan_config.pose_cost_metric = pose_cost_metric

        result = self.motion_gen.plan_single(start_joint_states, goal_pose_of_ee, plan_config)
        self._record_plan_diagnostics(result, target_pose, joint_pos, joint_vel, plan_config)
        return result
