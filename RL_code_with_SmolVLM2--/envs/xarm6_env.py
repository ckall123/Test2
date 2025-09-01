# =========================
# FILE: envs/xarm6_env.py
# =========================
from typing import Any, Dict, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from envs._ros_helpers import RosHelpers


from objects.spawner import Spawner
from vlm.sync_api import get_vlm_score



class XArm6GymEnv(gym.Env):
    """精簡版 xArm6 Gymnasium 環境（直接用 Gazebo/ROS 狀態，不做任何手算）。

    - action: 7 維連續（6 關節增量 + 1 夾爪增量），範圍 [-1, 1]
    - observation: Dict 包含 tf pose、joint、gripper、影像、旗標等
    - reward: 委託給外部 rewarder（傳入建構子）
    """

    metadata = {"render.modes": []}

    def __init__(self,
                 max_steps: int = 200,
                 arm_traj_topic: str = "/xarm6_traj_controller/joint_trajectory",
                 grip_action_name: str = "/xarm_gripper_traj_controller/follow_joint_trajectory",
                 gripper_joint_name: str = "drive_joint",
                 arm_step_rad: float = 0.20,
                 arm_limit_margin: float = 0.05,
                 arm_time_sec: float = 0.25,
                 grip_min: float = 0.0,
                 grip_max: float = 0.8552,
                 grip_step: float = 0.08,
                 grip_time_sec: float = 0.25,
                 joint_weights=(1, 1, 0.8, 0.6, 0.6, 0.6),
                 # 新增：影像/文字（先收參數，主要用 image）
                 vlm_interval: int = 999999,
                 vlm_prompt: str = "",
                 camera_topic: Optional[str] = None,
                 # reward 計算器
                 rewarder=None,
                 # frame 名稱
                 tcp_frame: str = "link_tcp",
                 world_frame: str = "world"):
        super().__init__()

        # 轉換 gripper action 名稱到 topic（若傳進來是 follow_joint_trajectory）
        grip_traj_topic = grip_action_name.replace("/follow_joint_trajectory", "/joint_trajectory")

        # ROS 幫手
        self.ros = RosHelpers(
            arm_traj_topic=arm_traj_topic,
            grip_traj_topic=grip_traj_topic,
            joint_names=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
            gripper_joint_name=gripper_joint_name,
            camera_topic=camera_topic,
            image_size=(224, 224),
        )

        # 參數
        self.max_steps = int(max_steps)
        self.arm_step = float(arm_step_rad)
        self.arm_limit_margin = float(arm_limit_margin)
        self.arm_time = float(arm_time_sec)
        self.grip_min = float(grip_min)
        self.grip_max = float(grip_max)
        self.grip_step = float(grip_step)
        self.grip_time = float(grip_time_sec)
        self.joint_weights = np.array(joint_weights, dtype=np.float32)
        self.vlm_interval = int(vlm_interval)
        self.vlm_prompt = str(vlm_prompt)

        self.tcp_frame = tcp_frame
        self.world_frame = world_frame

        # 狀態
        self.step_count = 0
        self._last_arm_cmd = np.zeros(6, dtype=np.float32)
        self._last_grip_cmd = 0.0

        # 動作空間：[-1,1]^7
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)

        # 觀測空間：
        self.observation_space = spaces.Dict({
            "tcp_pose": spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32),
            "joint_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),
            "gripper_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            "traj_error": spaces.Discrete(2),
            "ee_low": spaces.Discrete(2),
            "image": spaces.Box(low=0, high=255, shape=(224, 224, 3), dtype=np.uint8),
        })

        # rewarder
        self.rewarder = rewarder

    # -------------------- Gym API --------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.step_count = 0
        self.ros.clear_violation()

        # 將手臂送回 Home（用現有 joint + 小幅回正，這裡僅示範，可替換成你固定的 home）
        current = self.ros.get_joint_positions()
        target = np.zeros(6, dtype=np.float32)
        blend = 0.5
        cmd = current * (1.0 - blend) + target * blend
        self.ros.send_arm_traj(cmd.tolist(), duration=max(0.5, self.arm_time))

        # 夾爪張開
        self.ros.send_gripper_traj(self.grip_min, duration=max(0.3, self.grip_time))

        # 讓 ROS 跑一下
        for _ in range(20):
            self.ros.spin_once(0.02)

        obs = self._collect_obs()
        info = {}
        return obs, info

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        # ---- 解析動作 ----
        arm_delta = action[:6] * self.arm_step
        grip_delta = float(action[6]) * self.grip_step

        # ---- 讀目前狀態 ----
        q_now = self.ros.get_joint_positions()
        g_now = self.ros.get_gripper_pos()

        # ---- 產生/發布命令 ----
        q_cmd = q_now + arm_delta
        g_cmd = float(np.clip(g_now + grip_delta, self.grip_min, self.grip_max))

        self.ros.send_arm_traj(q_cmd.tolist(), duration=self.arm_time)
        self.ros.send_gripper_traj(g_cmd, duration=self.grip_time)

        # ---- 讓 Gazebo/ROS 前進一點 ----
        spin_iters = int(max(self.arm_time, self.grip_time) / 0.02) + 1
        for _ in range(spin_iters):
            self.ros.spin_once(0.02)

        self.step_count += 1

        # ---- 蒐集觀測 ----
        obs = self._collect_obs()

        # ---- 終止條件 ----
        terminated = bool(obs["traj_error"])  # 只要控制器報錯就結束（交給外部自動 reset）
        truncated = self.step_count >= self.max_steps

        # ---- 奬懲（委外給 rewarder）----
        if self.rewarder is not None:
            reward = float(self.rewarder.compute(obs=obs, action=action))
        else:
            reward = 0.0

        info: Dict[str, Any] = {}
        return obs, reward, terminated, truncated, info

    def render(self):
        return None

    def close(self):
        try:
            self.ros.node.destroy_node()
        except Exception:
            pass

    # -------------------- Helpers --------------------
    def _collect_obs(self) -> Dict[str, Any]:
        # TF: world -> tcp
        tcp_pose = self.ros.get_tf_pose(self.tcp_frame, source_frame=self.world_frame, timeout=0.05)
        if tcp_pose is None:
            tcp = np.zeros(7, dtype=np.float32)
        else:
            tcp = np.concatenate([tcp_pose.xyz, tcp_pose.quat], axis=0).astype(np.float32)

        # flags
        traj_error = 1 if self.ros.violation_flag else 0
        ee_low = 1 if (tcp[2] < 0.95) else 0  # <== 依你的 world/table 設定自行調整（例：桌面 1.0m）

        obs = {
            "tcp_pose": tcp,
            "joint_pos": self.ros.get_joint_positions().astype(np.float32),
            "gripper_pos": np.array([self.ros.get_gripper_pos()], dtype=np.float32),
            "traj_error": int(traj_error),
            "ee_low": int(ee_low),
            "image": self.ros.get_image(fill_if_none=True),
        }
        return obs
