# xarm6_gym_env.py
# ✅ 改寫目標：符合「關節為主」版完整規格 + 整合 Spawner

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List

from camera import TopDownCamera
from moveit_controller import MoveItController, ARM_JOINT_NAMES
from pose_tracker import PoseTracker, load_target_names
from utils import ActionHistory, preprocess_image
from gripper_contact import ContactMonitor
from attach_detach import AttachDetachClient
from spawner import Spawner   # ✅ 新增


@dataclass
class XArmEnvConfig:
    image_size: Tuple[int, int] = (96, 96)
    max_steps: int = 400
    action_scale: float = 0.08
    action_hist_len: int = 10
    x_range: Tuple[float, float] = (-1.05, 0.45)
    y_range: Tuple[float, float] = (-1.20, -0.40)
    z_height: float = 1.015
    close_thresh: float = 0.6
    open_thresh: float = 0.4
    contact_frames: int = 3
    v_thresh: float = 0.01
    envelope_dims: Tuple[float, float] = (0.06, 0.04)
    finger_link: str = "left_inner_finger"


class XArm6Env(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, node, executor, cfg: XArmEnvConfig = XArmEnvConfig()):
        super().__init__()
        self.node = node
        self.executor = executor
        self.cfg = cfg
        self.prev_img = None
        self.held_object = None

        # --- 模組 ---
        self.controller = MoveItController(node, executor)
        self.camera = TopDownCamera(node)
        self.tracker = PoseTracker(node)
        self.contact_monitor = ContactMonitor(node)
        self.attacher = AttachDetachClient(node)
        self.spawner = Spawner(node)  # ✅ 新增：Spawner 實例

        self.object_names = load_target_names()[:5]  # 固定 5 個
        self.k_objects = len(self.object_names)

        self.act_dim = len(ARM_JOINT_NAMES) + 1
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.act_dim,), dtype=np.float32)

        w, h = self.cfg.image_size
        image_shape = (h, w, 3)

        self._hist = ActionHistory(self.cfg.action_hist_len, self.act_dim)
        self.hist_dim = self.cfg.action_hist_len * self.act_dim
        self.state_dim = 6 + 1 + (self.k_objects * 5) + 1 + self.hist_dim

        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=255, shape=image_shape, dtype=np.uint8),
            "state": spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32),
        })

        self.step_count = 0
        self.node.get_logger().info("✅ XArm6Env 初始化完成")

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.node.get_logger().info("🔄 Reset environment")

        self.step_count = 0
        self._hist.clear()
        self.held_object = None
        self.attacher.clear_all()
        self.contact_monitor.reset()

        # ✅ 使用 Spawner 重生物件
        self.node.get_logger().info("🧩 Respawning tabletop objects...")
        self.spawner.delete_all()
        self.spawner.spawn(self.object_names)

        # 等待 PoseTracker 更新物件資訊
        for _ in range(30):
            objs = self.tracker.get_object_states(self.object_names)
            if len(objs) == self.k_objects:
                break
            self.executor.spin_once(timeout_sec=0.1)

        self.controller.go_home()
        self.controller.move_gripper(1.0)

        img = self._grab_image()
        self.prev_img = img

        obs = self._get_obs(img, contact_flag=0.0)
        info = {"image": img, "next_image": img}
        return obs, info

    def step(self, action: np.ndarray):
        self.step_count += 1
        dq = action[:6] * self.cfg.action_scale
        grip_cmd = float(np.clip(action[6], -1.0, 1.0))
        open_frac = (grip_cmd + 1.0) * 0.5

        q_now = self.controller.get_joint_positions()
        q_tar = q_now + dq
        self.controller.plan_and_execute(q_tar, open_frac)

        for _ in range(3):
            self.executor.spin_once(timeout_sec=0.01)

        self._hist.add(action)

        # ✅ 接觸與吸附判斷
        contact_ok, candidate = self.contact_monitor.check_dual_contact(self.object_names)
        contact_flag = float(contact_ok)

        if self.held_object and grip_cmd < self.cfg.open_thresh:
            model, link = self._parse_model_link(self.held_object)
            self.attacher.detach(model, link)
            self.held_object = None

        elif not self.held_object and grip_cmd > self.cfg.close_thresh and contact_ok:
            cand2 = self.contact_monitor.candidate(self.object_names)
            candidate = cand2 or candidate
            if candidate and self.contact_monitor.is_stable(
                candidate, self.cfg.contact_frames, self.cfg.v_thresh, self.cfg.envelope_dims
            ):
                model, link = self._parse_model_link(candidate)
                self.attacher.attach(model, link, self.cfg.finger_link)
                self.held_object = f"{model}::{link}"

        img = self._grab_image()
        obs = self._get_obs(img, contact_flag)

        reward = 0.0
        terminated = False
        truncated = self.step_count >= self.cfg.max_steps
        info = {"image": self.prev_img, "next_image": img}

        self.prev_img = img
        return obs, reward, terminated, truncated, info

    def render(self) -> np.ndarray:
        return self._grab_image()

    def close(self):
        self.node.get_logger().info("🧹 Environment closed")

    def _grab_image(self) -> np.ndarray:
        return self.camera.get_latest_frame(self.executor)

    def _parse_model_link(self, s: str) -> Tuple[str, str]:
        if "::" in s:
            model, link = s.split("::", 1)
            return model, link
        else:
            return s, "link"

    def _get_obs(self, img: np.ndarray, contact_flag: float) -> Dict[str, Any]:
        image_obs = preprocess_image(img, self.cfg.image_size)
        state_vec = self._build_state(contact_flag)
        return {"image": image_obs.astype(np.uint8), "state": state_vec.astype(np.float32)}

    def _build_state(self, contact_flag: float) -> np.ndarray:
        q = self.controller.get_joint_positions()
        q_limits = self.controller.joint_limits
        q_norm = np.clip(2 * (q - q_limits[:, 0]) / (q_limits[:, 1] - q_limits[:, 0]) - 1.0, -1.0, 1.0)

        g = float(self.controller.get_gripper_state())
        g_norm = np.array([g], dtype=np.float32)

        objs = self.tracker.get_object_states(self.object_names)
        obj_map = {o["name"]: (o["pos"], o["yaw"]) for o in objs}

        x_min, x_max = self.cfg.x_range
        y_min, y_max = self.cfg.y_range
        z_min, z_max = self.cfg.z_height - 0.1, self.cfg.z_height + 0.1

        obj_features = []
        for name in self.object_names:
            if name in obj_map:
                pos, yaw = obj_map[name]
                x = np.clip(2 * (pos[0] - x_min) / (x_max - x_min) - 1.0, -1.0, 1.0)
                y = np.clip(2 * (pos[1] - y_min) / (y_max - y_min) - 1.0, -1.0, 1.0)
                z = np.clip(2 * (pos[2] - z_min) / (z_max - z_min) - 1.0, 1.0)
                yaw_sin = np.sin(yaw)
                yaw_cos = np.cos(yaw)
                obj_features.append([x, y, z, yaw_sin, yaw_cos])
            else:
                obj_features.append([0.0, 0.0, 0.0, 0.0, 1.0])

        obj_flat = np.concatenate(obj_features, axis=0).astype(np.float32)
        hist = self._hist.vector()
        if hist.size == 0:
            hist = np.zeros(self.hist_dim, dtype=np.float32)

        contact = np.array([contact_flag], dtype=np.float32)
        return np.concatenate([q_norm, g_norm, obj_flat, contact, hist], axis=0)
