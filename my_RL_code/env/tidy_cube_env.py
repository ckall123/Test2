import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import rclpy

from env.camera import TopDownCamera
from env.fk import FKClient
from env.ik import IKClient  # ✅ 忘記的 IK 模組
from env.gripper import GripperStateTracker
from env.joint_controller import JointController
from object.spawner import Spawner  # ✅ 加入 Spawner
from env import reward  # ✅ 加入 reward 模組


class TidyCubeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, obs_type="pixels_agent_pos", resolution=(64, 64)):
        super().__init__()

        self.obs_type = obs_type
        self.render_mode = render_mode
        self.resolution = resolution

        # === ROS2 Init ===
        rclpy.init()

        # === 子模組初始化 ===
        self.camera = TopDownCamera(resolution=self.resolution)
        self.fk = FKClient()
        self.ik = IKClient()  # ✅ IK 初始化
        self.gripper = GripperStateTracker()
        self.controller = JointController()
        self.spawner = Spawner()  # ✅ 初始化 spawner

        # === Action: [dx, dy, dz, gripper] ===
        self.action_space = spaces.Box(
            low=np.array([-0.05, -0.05, -0.05, 0.0]),
            high=np.array([0.05, 0.05, 0.05, 1.0]),
            dtype=np.float32
        )

        # === Observation: image + ee pose + gripper ===
        image_shape = (self.resolution[1], self.resolution[0], 3)
        self.observation_space = spaces.Dict({
            "pixels": spaces.Box(low=0, high=255, shape=image_shape, dtype=np.uint8),
            "agent_pos": spaces.Box(
                low=np.array([-1, -1, -1, 0.0]),
                high=np.array([1, 1, 1, 1.0]),
                dtype=np.float32
            )
        })

    def _get_obs(self):
        img = self.camera.get_latest_frame()
        if img is None:
            rclpy.spin_once(self.camera, timeout_sec=0.1)
            img = self.camera.get_latest_frame()

        joint_msg = self.gripper.last_msg if hasattr(self.gripper, 'last_msg') else None
        if joint_msg is None:
            rclpy.spin_once(self.gripper, timeout_sec=0.1)
            joint_msg = self.gripper.last_msg if hasattr(self.gripper, 'last_msg') else None

        if joint_msg:
            joint_order = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
            name_to_pos = dict(zip(joint_msg.name, joint_msg.position))
            try:
                ordered_pos = [name_to_pos[name] for name in joint_order]
            except KeyError:
                ordered_pos = [0.0] * 6
        else:
            ordered_pos = [0.0] * 6

        pos = self.fk.compute_fk(['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6'], ordered_pos)
        gripper_val = self.gripper.get_state()

        obs = {
            "pixels": img,
            "agent_pos": np.concatenate([pos, [gripper_val]]).astype(np.float32)
        }
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        home = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.controller.send_joint_positions(home, duration=1.5)

        self.spawner.delete_all()
        self.spawner.spawn_random_objects(count=3)

        return self._get_obs(), {}

    def step(self, action):
        # ✅ IK 控制動起來！
        current = self._get_obs()["agent_pos"][:3]
        target = current + action[:3]
        angles = self.ik.compute_ik(target)
        if angles is not None:
            self.controller.send_joint_positions(angles, duration=1.0)

        self.gripper.gripper_state = float(action[3])  # 暫存更新（訓練用）

        obs = self._get_obs()
        reward_val = reward.reward_horizontal_alignment(obs["pixels"])
        terminated = False
        truncated = False
        info = {}
        return obs, reward_val, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            cv2.imshow("TidyCubeEnv", self.camera.get_latest_frame())
            cv2.waitKey(int(1000 / self.metadata["render_fps"]))
        elif self.render_mode == "rgb_array":
            return self.camera.get_latest_frame()

    def close(self):
        cv2.destroyAllWindows()
        self.camera.destroy_node()
        self.gripper.destroy_node()
        self.fk.destroy_node()
        self.ik.destroy_node()
        self.controller.destroy_node()
        self.spawner.destroy_node()
        rclpy.shutdown()
