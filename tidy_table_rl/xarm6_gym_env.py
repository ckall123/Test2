#!/usr/bin/env python3
"""
XArm6 Gym Environment (RL-VLM-F ready)
--------------------------------------
- Joint-space control (6 joints + 1 grip_cmd)
- Fixed state spec with HOLD flag and 5×K object features
- Action history compressed to 5D per step: [dx, dy, dz, dyaw, grip_cmd]
- Gazebo Classic link-attacher integration via Attach/Detach client
- Contact gating (multi-frame + envelope + velocity) delegated to ContactMonitor

Design goals
- Low coupling: controller/camera/tracker/attacher/contacts are thin dependencies
- Clear interfaces: this env only calls public methods, no internal constants
- Deterministic shapes: observation_space is fixed at __init__
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# --- External modules (thin interfaces expected) ---
from camera import TopDownCamera
from moveit_controller import MoveItController, ARM_JOINT_NAMES  # controller exposes joint_limits & get_ee_pose()
from pose_tracker import PoseTracker, load_target_names
from utils import ActionHistory, preprocess_image

# Optional modules provided by your project
from gripper_contact import ContactMonitor  # must offer: check_dual_contact(names) -> (ok:bool, candidate:dict|None)
from attach_detach import AttachDetachClient  # must offer: attach(model, link, finger_link), detach(model, link)


# =============================================================================
# Configuration
# =============================================================================
@dataclass
class XArmEnvConfig:
    # Image
    image_size: Tuple[int, int] = (96, 96)  # (W, H)

    # Episode
    max_steps: int = 400

    # Actions
    action_scale: float = 0.08  # per-step delta for joint increments
    use_joint_velocity: bool = False  # if True, include dq in state

    # Action history (compressed 5D per step)
    action_hist_len: int = 10

    # Workspace bounds for normalization (meters, world frame)
    x_range: Tuple[float, float] = (-0.30, 0.30)
    y_range: Tuple[float, float] = (0.10, 0.70)
    z_height: float = 0.25  # nominal table-top height; z will be normalized within ±0.10m around this

    # Objects (fixed K to keep state_dim constant)
    k_objects: Optional[int] = None  # if None, inferred from initial target list

    # Gripper + attach/detach policy
    close_thresh: float = 0.60  # grip_cmd > close_thresh => closing/closed
    open_thresh: float = 0.40   # grip_cmd < open_thresh  => opening/open
    contact_frames: int = 3     # consecutive frames of valid contact needed to attach
    v_thresh: float = 0.05      # max relative speed (m/s) to regard contact as stable
    envelope_dims: Tuple[float, float] = (0.06, 0.04)  # (width, depth) grasp envelope around finger tool frame

    # Attacher configuration
    finger_link: str = "gripper_tool_link"  # the link used to attach objects (tool center / finger pad)


# =============================================================================
# Environment
# =============================================================================
class XArm6Env(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, node, executor, cfg: XArmEnvConfig = XArmEnvConfig(), spawner=None):
        super().__init__()
        self.node = node
        self.executor = executor
        self.cfg = cfg
        self.spawner = spawner  # optional scene spawner with delete_all()/spawn_random_objects(k)

        # Modules
        self.controller = MoveItController(node, executor)
        self.camera = TopDownCamera(node)
        self.tracker = PoseTracker(node)
        self.contact_monitor = ContactMonitor(node)
        self.attacher = AttachDetachClient(node)

        # Objects (fix K at init to keep state_dim constant)
        init_names = load_target_names()
        self.object_names: List[str] = init_names
        self.K = cfg.k_objects if cfg.k_objects is not None else len(init_names)
        if self.K != len(init_names):
            # if K provided, ensure tracker/spawner will produce exactly K names later
            self.object_names = init_names[: self.K]

        # Actions: 6 joint increments + 1 grip_cmd in [-1, 1]
        self.act_dim = len(ARM_JOINT_NAMES) + 1
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.act_dim,), dtype=np.float32)

        # Compressed action history: 5 dims per step
        self.hist_dim = 5 * self.cfg.action_hist_len
        self._hist = ActionHistory(self.cfg.action_hist_len, 5)

        # State dimension = joints (+vel) + HOLD + 5*K objects + 5H history
        base_joints = 6 + (6 if self.cfg.use_joint_velocity else 0)
        self.state_dim = base_joints + 1 + (5 * self.K) + self.hist_dim

        # Observation space (image is H×W×3 uint8 for easy logging/storage)
        W, H = self.cfg.image_size
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(low=0, high=255, shape=(H, W, 3), dtype=np.uint8),
                "state": spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32),
            }
        )

        # Running episode state
        self.step_count = 0
        self.prev_img: Optional[np.ndarray] = None
        self.hold: int = 0  # 0/1
        self.held_object: Optional[str] = None
        self._last_ee_pose: Optional[Tuple[float, float, float, float]] = None  # (x,y,z,yaw)

        self.node.get_logger().info(
            f"✅ XArm6Env ready: K={self.K}, state_dim={self.state_dim}, hist_dim={self.hist_dim}"
        )

    # ---------------------------------------------------------------------
    # Gym API
    # ---------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.node.get_logger().info("🔄 Reset environment")
        self.step_count = 0
        self._hist.clear()
        self.hold = 0
        self.held_object = None

        # (Optional) Rebuild scene to guarantee fixed K
        if self.spawner is not None:
            try:
                self.spawner.delete_all()
                self.spawner.spawn_random_objects(self.K)
                # refresh ordered names from tracker/spawner to keep consistent mapping
                names = load_target_names()
                self.object_names = names[: self.K]
            except Exception as e:
                self.node.get_logger().warn(f"[reset] spawner failed: {e}")

        # Robot to home, open gripper
        self.controller.go_home()
        self.controller.move_gripper(1.0)  # fully open for safety

        # Init EE pose ref for action history compression
        ee = self.controller.get_ee_pose()  # (x,y,z,yaw)
        self._last_ee_pose = ee

        # Prime contacts/attacher state
        self.attacher.clear_all()

        # First image
        full_img = self._grab_image()
        self.prev_img = full_img

        obs = self._get_obs(full_img)
        info = {"image": full_img, "next_image": full_img}
        return obs, info

    def step(self, action: np.ndarray):
        self.step_count += 1

        # 1) Apply action
        grip_cmd = float(action[-1])  # [-1,1]
        dq = np.asarray(action[:6], dtype=np.float32) * self.cfg.action_scale
        self._apply_joint_step(dq, grip_cmd)

        # 2) Update compressed action history (based on EE delta)
        ee_now = self.controller.get_ee_pose()  # (x,y,z,yaw)
        dx, dy, dz, dyaw = self._ee_delta(self._last_ee_pose, ee_now)
        self._last_ee_pose = ee_now
        self._hist.add(np.array([dx, dy, dz, dyaw, grip_cmd], dtype=np.float32))

        # 3) Attach/Detach state machine
        self._grip_fsm(grip_cmd)

        # 4) Build obs/info (reward=0, relabeled later)
        full_img = self._grab_image()
        obs = self._get_obs(full_img)

        reward = 0.0
        terminated = False
        truncated = self.step_count >= self.cfg.max_steps
        info = {"image": self.prev_img, "next_image": full_img}
        self.prev_img = full_img

        return obs, reward, terminated, truncated, info

    def render(self) -> np.ndarray:
        return self._grab_image()

    def close(self):
        self.node.get_logger().info("🧹 Environment closed")

    # ---------------------------------------------------------------------
    # Internals
    # ---------------------------------------------------------------------
    def _apply_joint_step(self, dq: np.ndarray, grip_cmd: float):
        """Send a small joint increment + gripper command to the controller.
        grip_cmd∈[-1,1] is mapped to [0..1] opening fraction for the physical gripper.
        HOLD flag is managed separately by the attach/detach FSM.
        """
        try:
            q_now = self.controller.get_joint_positions()
            q_tar = q_now + dq
            g_open_frac = float(np.clip(0.5 * (grip_cmd + 1.0), 0.0, 1.0))
            self.controller.plan_and_execute(q_tar, g_open_frac)
        except Exception as e:
            self.node.get_logger().warn(f"[apply_joint_step] failed: {e}")
        finally:
            # Let ROS2 spin a bit
            for _ in range(3):
                self.executor.spin_once(timeout_sec=0.01)

    def _grab_image(self) -> np.ndarray:
        return self.camera.get_latest_frame(self.executor)

    def _get_obs(self, img: np.ndarray) -> Dict[str, Any]:
        W, H = self.cfg.image_size
        image_obs = preprocess_image(img, (W, H)).astype(np.uint8)
        state_vec = self._build_state()
        return {"image": image_obs, "state": state_vec.astype(np.float32)}

    def _build_state(self) -> np.ndarray:
        # --- Joints (normalized to [-1,1]) ---
        q = self.controller.get_joint_positions().astype(np.float32)
        q_limits = self.controller.joint_limits.astype(np.float32)  # shape: (6, 2)
        q_norm = 2.0 * (q - q_limits[:, 0]) / (q_limits[:, 1] - q_limits[:, 0]) - 1.0
        q_feats = [np.clip(q_norm, -1.0, 1.0)]

        if self.cfg.use_joint_velocity:
            try:
                dq = self.controller.get_joint_velocities().astype(np.float32)
                # If velocity limits available on controller, normalize similarly; else tanh clip
                if hasattr(self.controller, "joint_velocity_limits"):
                    vlim = self.controller.joint_velocity_limits.astype(np.float32)  # (6,)
                    dq_norm = np.clip(dq / np.maximum(vlim, 1e-6), -1.0, 1.0)
                else:
                    dq_norm = np.tanh(dq)
                q_feats.append(dq_norm)
            except Exception as e:
                self.node.get_logger().warn(f"[build_state] velocity fetch failed: {e}")
                q_feats.append(np.zeros(6, dtype=np.float32))

        # --- HOLD flag (1 dim) ---
        hold_arr = np.array([float(self.hold)], dtype=np.float32)

        # --- Objects: 5 dims each (x, y, z, sin(yaw), cos(yaw)) ---
        obj_flat = self._object_features_5k()

        # --- Action history (5H) ---
        hist = self._hist.vector()
        if hist.size == 0:
            hist = np.zeros(self.hist_dim, dtype=np.float32)

        return np.concatenate(q_feats + [hold_arr, obj_flat, hist], axis=0)

    def _object_features_5k(self) -> np.ndarray:
        x_min, x_max = self.cfg.x_range
        y_min, y_max = self.cfg.y_range
        z_center = self.cfg.z_height
        z_min, z_max = z_center - 0.10, z_center + 0.10

        # Get tracker outputs and map by name
        objs = self.tracker.get_object_states(self.object_names)  # list of {name, pos(np3), yaw(float)}
        obj_map = {o["name"]: (o["pos"], o["yaw"]) for o in objs}

        feats: List[List[float]] = []
        for idx in range(self.K):
            name = self.object_names[idx] if idx < len(self.object_names) else None
            if name is not None and name in obj_map:
                pos, yaw = obj_map[name]
                x = np.clip(2.0 * (pos[0] - x_min) / (x_max - x_min) - 1.0, -1.0, 1.0)
                y = np.clip(2.0 * (pos[1] - y_min) / (y_max - y_min) - 1.0, -1.0, 1.0)
                z = np.clip(2.0 * (pos[2] - z_min) / (z_max - z_min) - 1.0, -1.0, 1.0)
                feats.append([x, y, z, float(np.sin(yaw)), float(np.cos(yaw))])
            else:
                # Default neutral feature if missing
                feats.append([0.0, 0.0, 0.0, 0.0, 1.0])

        return np.asarray(feats, dtype=np.float32).reshape(-1)

    # ---------------------------------------------------------------------
    # Grip FSM + contact gating
    # ---------------------------------------------------------------------
    def _grip_fsm(self, grip_cmd: float) -> None:
        """Attach/Detach policy driven by grip_cmd and contact gating.
        - Attach when grip_cmd>close_thresh AND stable contact candidate exists
        - Detach when grip_cmd<open_thresh
        """
        # Detach condition first for responsiveness
        if self.hold == 1 and grip_cmd < self.cfg.open_thresh:
            if self.held_object is not None:
                try:
                    model, link = self.held_object.split("::", 1)
                    self.attacher.detach(model, link)
                except Exception as e:
                    self.node.get_logger().warn(f"[detach] failed: {e}")
            self.hold = 0
            self.held_object = None
            return

        # Attach condition
        if self.hold == 0 and grip_cmd > self.cfg.close_thresh:
            ok, cand = self.contact_monitor.check_dual_contact(self.object_names)
            if not ok or cand is None:
                return

            # Optional gating: envelope and relative velocity (delegated to ContactMonitor if supported)
            stable = True
            if hasattr(self.contact_monitor, "is_stable"):
                stable = self.contact_monitor.is_stable(
                    cand,
                    frames=self.cfg.contact_frames,
                    v_thresh=self.cfg.v_thresh,
                    envelope=self.cfg.envelope_dims,
                )
            if not stable:
                return

            # Execute attach
            try:
                model = cand.get("model", "")
                link = cand.get("link", "")
                self.attacher.attach(model, link, self.cfg.finger_link)
                self.hold = 1
                self.held_object = f"{model}::{link}"
            except Exception as e:
                self.node.get_logger().warn(f"[attach] failed: {e}")

    # ---------------------------------------------------------------------
    # Utilities
    # ---------------------------------------------------------------------
    @staticmethod
    def _ee_delta(prev_pose: Optional[Tuple[float, float, float, float]],
                  now_pose: Optional[Tuple[float, float, float, float]]):
        if prev_pose is None or now_pose is None:
            return 0.0, 0.0, 0.0, 0.0
        px, py, pz, pyaw = prev_pose
        nx, ny, nz, nyaw = now_pose
        dyaw = nyaw - pyaw
        # wrap to [-pi, pi]
        dyaw = (dyaw + np.pi) % (2 * np.pi) - np.pi
        return float(nx - px), float(ny - py), float(nz - pz), float(dyaw)
