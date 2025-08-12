import gymnasium as gym
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from sim_object import SimObject
from spawner import randomize_objects

from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import TransformStamped

from buffers.image_buffer import ImageBuffer
from gymnasium.spaces import Box, Dict as SpaceDict

from vlm.vlm_interface import get_vlm_score

import cv2
from cv_bridge import CvBridge

IMG_H, IMG_W = 64, 64

class XArm6GymEnv(gym.Env):
    """
    ROS + Gym 環境整合的 xArm6 機械手臂環境。
    ✨ 功能：支援 tf 正向運動學、碰撞偵測、觀察標準化、圖像紀錄、偏好學習用圖片收集。
    """

    metadata = {"render_modes": []}

    _LIMITS = np.array([
        [-3.1067,  3.1067],
        [-2.0595,  2.0944],
        [-3.1067,  0.1919],
        [-3.1067,  3.1067],
        [-1.6929,  3.1067],
        [-3.1067,  3.1067],
    ], dtype=np.float32)

    def __init__(self, fk_mode="tf", enable_collision=False, table_height=0.0, max_steps=200):
        super().__init__()

        self.fk_mode = fk_mode
        self.enable_collision = enable_collision
        self.table_height = table_height
        self.max_steps = max_steps
        self.step_count = 0

        # ROS Node 初始化（由 train.py 保證 rclpy.init()）
        self.node: Node = Node("xarm6_gym_env")

        # ROS Publisher / Subscriber
        self.cmd_pub = self.node.create_publisher(JointTrajectory, "/xarm6_traj_controller/joint_trajectory", 10)
        self.state_sub = self.node.create_subscription(JointState, "/joint_states", self._joint_state_cb, 10)
        self.image_sub = self.node.create_subscription(Image, "/camera/image_raw", self._image_cb, 10)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self.node)

        self.bridge = CvBridge()
        self.latest_image = None
        self.image_buffer = ImageBuffer()

        # Gym 定義 action / observation space
        self.action_space = gym.spaces.Box(
            low=np.array([-1] * 6 + [0.0], dtype=np.float32),
            high=np.array([1] * 6 + [0.8552], dtype=np.float32),
        )

        self.objects = []  # 延後在 reset 時產生

        obs_dim = 7 + 3 * 2  # joints + gripper + 2 objects (xyz)
        
        self.observation_space = SpaceDict({
            "state": Box(low=-1.0, high=1.0, shape=(7 + 3*2,), dtype=np.float32),
            "image": Box(low=0, high=255, shape=(IMG_H, IMG_W, 3), dtype=np.uint8),
        })

        self.current_joint_state = np.zeros(6, dtype=np.float32)
        self.gripper_angle: float = 0.0
        self.attached_object: SimObject | None = None

    def _image_cb(self, msg: Image):
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception:
            self.latest_image = None

    def _joint_state_cb(self, msg: JointState):
        self.current_joint_state = np.array(msg.position[:6], dtype=np.float32)

    def _compute_ee_position(self, joint_angles: np.ndarray) -> np.ndarray:
        if self.fk_mode == "tf":
            try:
                now = rclpy.time.Time()
                transform: TransformStamped = self.tf_buffer.lookup_transform(
                    "base_link", "ee_link", now, timeout=rclpy.duration.Duration(seconds=0.1)
                )
                pos = transform.transform.translation
                return np.array([pos.x, pos.y, pos.z], dtype=np.float32)
            except Exception:
                pass

        x = 0.3 + 0.2 * np.cos(joint_angles[0]) + 0.2 * np.cos(joint_angles[0] + joint_angles[1])
        y = 0.0 + 0.2 * np.sin(joint_angles[0]) + 0.2 * np.sin(joint_angles[0] + joint_angles[1])
        z = 0.05 + 0.1 * (joint_angles[2] + 1.5)
        return np.array([x, y, max(0.0, z)], dtype=np.float32)

    def attach_object(self, obj: SimObject):
        self.attached_object = obj
        obj.attached = True

    def detach_object(self):
        if self.attached_object:
            self.attached_object.attached = False
            self.attached_object = None

    def _normalize_obs(self, raw_obs: np.ndarray) -> np.ndarray:
        joint_scaled = 2 * (self.current_joint_state - self._LIMITS[:, 0]) / (self._LIMITS[:, 1] - self._LIMITS[:, 0]) - 1
        grip_scaled = 2 * self.gripper_angle / 0.8552 - 1
        object_scaled = raw_obs[7:] / 1.0
        return np.concatenate([joint_scaled, [grip_scaled], object_scaled])

    def _check_invalid_state(self) -> bool:
        ee_pos = self._compute_ee_position(self.current_joint_state)
        return ee_pos[2] < self.table_height or np.any(np.isnan(self.current_joint_state))

    def _get_obs(self):
        raw_obs = list(self.current_joint_state) + [self.gripper_angle]
        for obj in self.objects:
            raw_obs.extend(obj.position)
        state_vec = self._normalize_obs(np.asarray(raw_obs, dtype=np.float32))  # ← 你原本就有
        # 影像：若未就緒先給零
        if self.latest_image is None:
            img = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
        else:
            img = cv2.resize(self.latest_image, (IMG_W, IMG_H))
        return {"state": state_vec, "image": img}

    def _geom_layout_score(self):
        # 例：對齊 + 間距一致性（用物體 pose 算，不用影像）
        if len(self.objects) < 2: return 0.0
        P = np.stack([o.position for o in self.objects], axis=0)
        y_std = np.std(P[:,1])
        xs = np.sort(P[:,0])
        gaps = np.diff(xs)
        gap_var = np.var(gaps) if len(gaps)>0 else 0.0
        score_align   = np.exp(-5.0*y_std)
        score_spacing = np.exp(-10.0*gap_var)
        return 0.5*score_align + 0.5*score_spacing

    def _compute_reward(self, obs: dict) -> float:
        # 你原本的幾何 shaping（用 state 向量做）
        state = obs["state"]
        reward_geom = 0.0
        ee_pos = self._compute_ee_position(self.current_joint_state)
        object_positions = state[7:].reshape(-1, 3)
        dists = np.linalg.norm(object_positions - ee_pos, axis=1)
        reward_geom += max(0.0, 1.0 - dists.min() / 0.5)
        if self.attached_object:
            reward_geom += 2.0
            if self.attached_object.position[2] > 0.15:
                reward_geom += 1.0
        if self.enable_collision and self._check_invalid_state():
            reward_geom -= 1.0

        # 版面整齊幾何分
        reward_layout = self._geom_layout_score()

        # VLM 美感分（用最新影像；可再做EMA平滑/差分增益）
        reward_vlm = get_vlm_score(obs["image"])

        # 合成：自己調 αβγ 權重
        return 0.4*reward_geom + 0.3*reward_layout + 0.3*reward_vlm

    def step(self, action: np.ndarray):
        self.step_count += 1
        self.gripper_angle = np.clip(self.gripper_angle + action[6], 0.0, 0.8552)

        traj = JointTrajectory()
        traj.joint_names = [f"joint{i+1}" for i in range(6)]
        point = JointTrajectoryPoint()
        step_scale = 0.05 #  step() 直接 current_joint_state + action[:6]，建議乘個係數（例如 0.05），不然學習會抖
        point.positions = (self.current_joint_state + step_scale * action[:6]).tolist()
        point.time_from_start.sec = 0
        point.time_from_start.nanosec = 50_000_000
        traj.points.append(point)
        self.cmd_pub.publish(traj)
        rclpy.spin_once(self.node, timeout_sec=0.05)

        ee_pos = self._compute_ee_position(self.current_joint_state)

        if self.gripper_angle < 0.02:
            self.detach_object()
        elif self.gripper_angle > 0.6 and self.attached_object is None:
            for obj in self.objects:
                if np.linalg.norm(ee_pos - obj.position) < 0.05:
                    self.attach_object(obj)
                    break

        if self.attached_object:
            self.attached_object.position = ee_pos + np.array([0, 0, -0.05], dtype=np.float32)

        obs = self._get_obs()
        reward = self._compute_reward(obs)
        done = self._check_invalid_state() or (self.attached_object and self.attached_object.position[2] > 0.2)
        truncated = self.step_count >= self.max_steps

        if self.latest_image is not None:
            self.image_buffer.add(
                episode_id=0,
                step_index=self.step_count,
                image=self.latest_image.copy(),
                metadata={"obs": obs}
            )

        info = {
            "ee_pos": ee_pos.tolist(),
            "gripper": self.gripper_angle,
            "attached": self.attached_object.name if self.attached_object else None,
        }
        return obs, reward, done, truncated, info

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.step_count = 0
        self.current_joint_state[:] = 0.0
        self.gripper_angle = 0.0
        self.detach_object()

        self.objects = randomize_objects()  # spawn 物件延後到此處
        obs = self._get_obs()
        return obs, {}

    def close(self):
        if rclpy.ok():  # ✅ 只在 ROS 還活著時才 shutdown
            self.node.destroy_node()
            rclpy.shutdown()

