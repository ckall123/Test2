#!/usr/bin/env python3
"""
xarm6_gym_env.py (RL-VLM-F 核心環境)

職責：
1. 實現 Gym API：reset / step / render
2. 負責與 ROS2 各模組整合（MoveIt, Camera, PoseTracker）
3. 不計算 reward（由 reward model relabel）
4. 回傳可用於 ReplayBuffer 的 (obs, info)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List

from camera import TopDownCamera
from moveit_controller import MoveItController, ARM_JOINT_NAMES
from pose_tracker import PoseTracker, load_target_names
from utils import ActionHistory, preprocess_image


@dataclass
class XArmEnvConfig:
    image_size: Tuple[int, int] = (96, 96)
    max_steps: int = 400
    action_scale: float = 0.08
    action_hist_len: int = 10


class XArm6Env(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, node, executor, cfg: XArmEnvConfig = XArmEnvConfig()):
        super().__init__()
        self.node = node
        self.executor = executor
        self.cfg = cfg
        self.prev_img = None

        # --- 模組初始化 ---
        self.controller = MoveItController(node, executor)
        self.camera = TopDownCamera(node)
        self.tracker = PoseTracker(node)
        self.object_names = load_target_names()

        # --- 動作空間（6 joints + 1 gripper） ---
        self.act_dim = len(ARM_JOINT_NAMES) + 1
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.act_dim,), dtype=np.float32)

        # --- 觀測空間 ---
        w, h = self.cfg.image_size
        image_shape = (h, w, 3)

        self.action_hist = ActionHistory(self.cfg.action_hist_len, self.act_dim)
        self.hist_dim = self.cfg.action_hist_len * self.act_dim
        self.state_dim = 6 + 1 + (len(self.object_names) * 4) + self.hist_dim

        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=255, shape=image_shape, dtype=np.uint8),
            "state": spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32),
        })

        self.step_count = 0
        self.node.get_logger().info("✅ XArm6Env 初始化完成")

    # =====================================================
    # ✅ Gym API
    # =====================================================

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.node.get_logger().info("🔄 Reset environment")
        self.step_count = 0
        self.action_hist.clear()

        self.controller.go_home()
        self.controller.move_gripper(0.85)

        full_img = self._grab_image()
        self.prev_img = full_img  # ✅ 儲存初始畫面

        obs = self._get_obs(full_img)
        info = {
            "image": full_img,
            "next_image": full_img  # ✅ 初始 next_image 為同一張
        }
        return obs, info

    def step(self, action: np.ndarray):
        self.step_count += 1
        self._apply_action(action)
        self.action_hist.add(action)

        full_img = self._grab_image()
        obs = self._get_obs(full_img)

        reward = 0.0
        done = False
        truncated = self.step_count >= self.cfg.max_steps

        info = {
            "image": self.prev_img,       # ✅ 上一步影像
            "next_image": full_img        # ✅ 當前影像
        }

        self.prev_img = full_img         # ✅ 更新 prev_img 為下一步用

        return obs, reward, done, truncated, info

    def render(self) -> np.ndarray:
        return self._grab_image()

    def close(self):
        self.node.get_logger().info("🧹 Environment closed")

    # =====================================================
    # 🔧 輔助函式
    # =====================================================

    def _apply_action(self, action: np.ndarray):
        a = np.asarray(action, dtype=np.float32)
        dq = a[:6] * self.cfg.action_scale
        q_now = self.controller.get_joint_positions()
        q_tar = q_now + dq
        g_val = float(np.clip((a[-1] + 1.0) * 0.5, 0.0, 1.0)) * 0.8552

        try:
            self.controller.plan_and_execute(q_tar, g_val)
        except Exception as e:
            self.node.get_logger().warn(f"[apply_action] failed: {e}")

        for _ in range(3):
            self.executor.spin_once(timeout_sec=0.01)

    def _grab_image(self) -> np.ndarray:
        return self.camera.get_latest_frame(self.executor)

    def _get_obs(self, img: np.ndarray) -> Dict[str, Any]:
        image_obs = preprocess_image(img, self.cfg.image_size)
        state_vec = self._build_state()
        return {"image": image_obs.astype(np.uint8), "state": state_vec.astype(np.float32)}

    def _build_state(self) -> np.ndarray:
        q = self.controller.get_joint_positions()
        g = np.array([self.controller.get_gripper_state()], dtype=np.float32)

        objs = self.tracker.get_object_states(self.object_names)
        obj_features = []
        for o in objs:
            obj_features.append(np.concatenate([o["pos"], [o["yaw"]]]))
        obj_flat = np.concatenate(obj_features, axis=0) if obj_features else np.zeros(len(self.object_names) * 4)

        hist = self.action_hist.vector()
        if hist.size == 0:
            hist = np.zeros(self.hist_dim, dtype=np.float32)

        return np.concatenate([q, g, obj_flat, hist], axis=0)
