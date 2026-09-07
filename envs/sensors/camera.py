from isaaclab.sensors import TiledCameraCfg, TiledCamera
import isaaclab.utils.math as math_utils
from isaaclab.utils import configclass
import isaaclab.sim.views.usd_frame_view as isaaclab_usd_frame_view
import omni.usd
import os
from pxr import UsdGeom

import numpy as np
import torch
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode
import warp as wp
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .._base_task import BaseTask
    from tacex_uipc import UipcInteractiveScene

@configclass
class CameraCfg(TiledCameraCfg):
    name: str = 'camera'
    look_at_target: tuple[float, float, float] | None = None
    # Render at an Isaac Sim 6-safe resolution, then expose the public
    # UniVTAC camera shape to collectors and policies.
    observation_resolution: tuple[int, int] | None = (480, 270)

class CameraManager:
    def __init__(self, cfg_list: list[CameraCfg], task:'BaseTask'):
        self.scene = task.scene
        self.task = task
        self.cfg_list = cfg_list
        self.cameras = {}
        # Camera-to-mount transforms come from each camera's authored local USD
        # transform.  The wrist hierarchy is static under the Isaac Lab 3
        # Fabric view, so its camera must be driven from articulation state.
        self._attached_camera_offsets = {}
        self._attached_camera_body_indices = {}

    @staticmethod
    def _local_camera_pose(prim, device: str):
        """Read a camera's authored local pose as Lab-order XYZW tensors."""
        local_transform = UsdGeom.Xformable(prim).GetLocalTransformation()
        position = local_transform.ExtractTranslation()
        orientation = local_transform.ExtractRotationQuat()
        imaginary = orientation.GetImaginary()
        return (
            torch.tensor([[position[0], position[1], position[2]]], dtype=torch.float32, device=device),
            torch.tensor(
                [[imaginary[0], imaginary[1], imaginary[2], orientation.GetReal()]],
                dtype=torch.float32,
                device=device,
            ),
        )

    def setup(self): 
        self.cameras = {
            cam_cfg.name: self.add_camera(cam_cfg) for cam_cfg in self.cfg_list
        }

    def add_camera(self, cam_cfg: CameraCfg):
        # Isaac Lab's FrameView requires a concrete USD prim path while it initializes a camera.
        # This validation workflow deliberately uses one environment, so resolve both pre-authored
        # (wrist) and spawned (head) camera paths to the one existing environment.
        if self.scene.cfg.num_envs == 1:
            cam_cfg.prim_path = cam_cfg.prim_path.replace(self.scene.env_regex_ns, self.scene.env_prim_paths[0])
        camera = TiledCamera(cam_cfg)
        prim = omni.usd.get_context().get_stage().GetPrimAtPath(cam_cfg.prim_path)
        if not prim.IsValid():
            raise RuntimeError(f"Camera prim was not found: {cam_cfg.prim_path}")

        # The authored wrist-camera scale is float3 while Isaac Lab 3's Fabric
        # bootstrap allocates a Vec3dArray.  Its stock USD view therefore
        # rejects an otherwise valid scale and the old workaround fell back to
        # static USD poses.  Patch get_scales only while this camera initializes
        # so Fabric receives the same numeric scale as float32 and can continue
        # providing the live articulated pose thereafter.  No USD asset,
        # transform value, physics state, or renderer setting is changed.
        scale_attr = prim.GetAttribute("xformOp:scale")
        use_usd_pose_view = scale_attr.IsValid() and str(scale_attr.GetTypeName()) != "double3"
        if use_usd_pose_view and os.environ.get("UNIVTAC_CAMERA_DIAGNOSTICS") == "1":
            layers = [str(spec.layer.identifier) for spec in scale_attr.GetPropertyStack()]
            def _xform_attributes(xform_prim):
                return {
                    attr.GetName(): str(attr.Get())
                    for attr in xform_prim.GetAttributes()
                    if attr.GetName().startswith("xformOp:")
                }
            print(
                "[CAMERA_DIAGNOSTIC]",
                f"prim={prim.GetPath()}",
                f"scale_type={scale_attr.GetTypeName()}",
                f"scale_value={scale_attr.Get()}",
                f"layers={layers}",
                f"local_xform={_xform_attributes(prim)}",
                f"parent={prim.GetParent().GetPath()}",
                f"parent_xform={_xform_attributes(prim.GetParent())}",
            )
        if use_usd_pose_view:
            original_get_scales = isaaclab_usd_frame_view.UsdFrameView.get_scales

            def get_scales_with_float3_support(frame_view, indices=None):
                indices_list = frame_view._resolve_indices(indices)
                scales = []
                for prim_idx in indices_list:
                    scale = np.asarray(
                        frame_view._prims[prim_idx].GetAttribute("xformOp:scale").Get(), dtype=np.float32
                    ).reshape(-1)
                    if scale.size == 1:
                        scale = np.repeat(scale, 3)
                    if scale.size != 3:
                        raise RuntimeError(f"Expected scalar or vec3 camera scale, got shape {scale.shape}")
                    scales.append(scale)
                return wp.array(np.asarray(scales, dtype=np.float32), dtype=wp.float32, device=frame_view._device)

            isaaclab_usd_frame_view.UsdFrameView.get_scales = get_scales_with_float3_support
        try:
            camera._initialize_impl()
        finally:
            if use_usd_pose_view:
                isaaclab_usd_frame_view.UsdFrameView.get_scales = original_get_scales
        camera._is_initialized = True
        if cam_cfg.look_at_target is not None:
            # Resolve the static head pose through Isaac Lab's own camera-frame
            # convention helper.  This avoids duplicating an OpenGL quaternion
            # conversion at the task boundary.
            camera.set_world_poses_from_view(
                torch.tensor([cam_cfg.offset.pos], dtype=torch.float32, device=camera._device),
                torch.tensor([cam_cfg.look_at_target], dtype=torch.float32, device=camera._device),
            )
        self.scene.sensors[f'camera_{cam_cfg.name}'] = camera
        if cam_cfg.spawn is None and "/WristCamera/" in cam_cfg.prim_path:
            if prim.GetParent().GetName() != "WristCamera":
                raise RuntimeError(
                    f"Expected {cam_cfg.prim_path} to be a direct child of WristCamera, "
                    f"found parent {prim.GetParent().GetPath()}."
                )
            # Fabric reports a stale authored world pose for this nested USD
            # camera while the articulation body already reports its live pose.
            # Combining those two frames produced a rigid but incorrect ~0.67 m
            # camera-to-hand displacement.  The USD child transform is the
            # authoritative camera-to-mount calibration.
            self._attached_camera_offsets[cam_cfg.name] = self._local_camera_pose(prim, camera._device)
            self._attached_camera_body_indices[cam_cfg.name] = None
        return camera

    @staticmethod
    def _as_torch(value):
        """Use a Torch view of Isaac Lab Fabric proxy arrays when available."""
        return value.torch if hasattr(value, "torch") else value

    def update_attached_camera_poses(self):
        """Synchronize pre-authored wrist cameras with the live hand pose.

        Isaac Lab's USD FrameView reads the authored wrist-camera transform but
        does not receive its articulated parent transform in this runtime.  The
        camera's authored local transform is composed with the live
        ``WristCamera`` body pose on every render.
        """
        if not self._attached_camera_offsets:
            return

        robot_manager = self.task._robot_manager
        robot_data = robot_manager.robot.data
        for name, offset in self._attached_camera_offsets.items():
            camera = self.cameras[name]
            body_idx = self._attached_camera_body_indices[name]
            if body_idx is None:
                body_ids, body_names = robot_manager.robot.find_bodies("WristCamera")
                if len(body_ids) != 1:
                    raise RuntimeError(
                        f"Expected exactly one WristCamera body for {name}, found {body_names}."
                    )
                body_idx = int(body_ids[0])
                self._attached_camera_body_indices[name] = body_idx
                if os.environ.get("UNIVTAC_CAMERA_DIAGNOSTICS") == "1":
                    print(f"[CAMERA_DIAGNOSTIC] {name} uses robot body {body_names[0]} index={body_idx}")
            mount_pos_w = self._as_torch(robot_data.body_link_pos_w)[:, body_idx]
            mount_quat_w = self._as_torch(robot_data.body_link_quat_w)[:, body_idx]
            camera_pos_w, camera_quat_w = math_utils.combine_frame_transforms(
                mount_pos_w, mount_quat_w, offset[0], offset[1]
            )
            # The captured offset is the authored USD/OpenGL camera frame,
            # hence preserve that convention when setting the live world pose.
            camera.set_world_poses(camera_pos_w, camera_quat_w, convention="opengl")
            # ``scene.update()`` otherwise reads the static USD hierarchy back
            # into CameraData.  Keep the externally reported pose consistent
            # with the live pose used by the renderer.
            camera._update_camera_state(
                pos_src=wp.from_torch(camera_pos_w.contiguous(), dtype=wp.vec3f),
                quat_src=wp.from_torch(camera_quat_w.contiguous(), dtype=wp.quatf),
                update_pose=True,
            )
            if camera._render_data is not None:
                camera._renderer.update_camera(
                    camera._render_data, camera._data.pos_w, camera._data.quat_w_world, camera._data.intrinsic_matrices
                )

    def get_attachment_diagnostics(self):
        """Return read-only pose checks for cameras mounted on articulation bodies."""
        diagnostics = {}
        robot_data = self.task._robot_manager.robot.data
        hand_idx = self.task._robot_manager._body_idx
        hand_pos_w = self._as_torch(robot_data.body_link_pos_w)[:, hand_idx]
        for name, offset in self._attached_camera_offsets.items():
            body_idx = self._attached_camera_body_indices[name]
            if body_idx is None:
                continue
            camera_pos_w = self._as_torch(self.cameras[name].data.pos_w)
            mount_pos_w = self._as_torch(robot_data.body_link_pos_w)[:, body_idx]
            diagnostics[name] = {
                "mount_body_index": int(body_idx),
                "authored_camera_to_mount_translation_m": offset[0][0].detach().cpu().tolist(),
                "authored_camera_to_mount_quaternion_xyzw": offset[1][0].detach().cpu().tolist(),
                "authored_camera_to_mount_distance_m": float(torch.linalg.norm(offset[0][0]).item()),
                "reported_camera_to_mount_distance_m": float(
                    torch.linalg.norm(camera_pos_w[0] - mount_pos_w[0]).item()
                ),
                "reported_camera_to_panda_hand_distance_m": float(
                    torch.linalg.norm(camera_pos_w[0] - hand_pos_w[0]).item()
                ),
            }
        return diagnostics

    @staticmethod
    def _resize_observation(value: torch.Tensor, resolution: tuple[int, int] | None, data_type: str):
        if resolution is None:
            return value
        target_width, target_height = resolution
        if value.shape[0] == target_height and value.shape[1] == target_width:
            return value
        if value.ndim == 2:
            channel_first = value.unsqueeze(0)
            channel_last = False
        elif value.ndim == 3:
            channel_first = value.permute(2, 0, 1)
            channel_last = True
        else:
            raise RuntimeError(f"Unsupported {data_type} camera output shape: {tuple(value.shape)}")
        if data_type == "depth":
            resized = F.resize(
                channel_first,
                [target_height, target_width],
                interpolation=InterpolationMode.NEAREST_EXACT,
            )
        else:
            resized = F.resize(
                channel_first,
                [target_height, target_width],
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )
        if channel_last:
            return resized.permute(1, 2, 0)
        return resized.squeeze(0)
    
    def get_observations(self, data_types: list[str] = None):
        obs = {}
        if data_types is None:
            data_types = ['rgb', 'rgba']
        for name, cam in self.cameras.items():
            obs[name] = {}
            for data_type in data_types:
                if data_type == 'rgb':
                    value = cam.data.output['rgb'].squeeze(0)
                    obs[name]['rgb'] = self._resize_observation(value, cam.cfg.observation_resolution, data_type)
                elif data_type == 'rgba':
                    value = cam.data.output['rgba'].squeeze(0)
                    obs[name]['rgba'] = self._resize_observation(value, cam.cfg.observation_resolution, data_type)
                elif data_type == 'depth':
                    value = cam.data.output['depth'].squeeze(0)
                    obs[name]['depth'] = self._resize_observation(value, cam.cfg.observation_resolution, data_type)
        return obs
