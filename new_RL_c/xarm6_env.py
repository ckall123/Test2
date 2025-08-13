import gymnasium as gym
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from sensor_msgs.msg import JointState, Image
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory

from gymnasium.spaces import Box, Dict as SpaceDict
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import TransformStamped

from vlm.vlm_interface import get_vlm_score

import cv2
from cv_bridge import CvBridge

IMG_H, IMG_W = 64, 64
WORKSPACE = {"x": (0.30, 0.80), "y": (-0.30, 0.30), "z": (0.00, 0.40)}

class XArm6GymEnv(gym.Env):
    _LIMITS = np.array([
        [-3.1067,  3.1067],
        [-2.0595,  2.0944],
        [-3.1067,  0.1919],
        [-3.1067,  3.1067],
        [-1.6929,  3.1067],
        [-3.1067,  3.1067],
    ], dtype=np.float32)

    def __init__(self, max_steps=200, vlm_interval=20, vlm_prompt=".", camera_topic="/camera/image_raw",
                 arm_traj_topic="/xarm6_traj_controller/joint_trajectory",
                 grip_action_name="/xarm_gripper_traj_controller/follow_joint_trajectory",
                 gripper_joint_name="drive_joint",
                 # 旋鈕
                 arm_step_rad=0.20, arm_limit_margin=0.05, arm_time_sec=0.25,
                 grip_min=0.0, grip_max=0.8552, grip_step=0.08, grip_time_sec=0.25,
                 joint_weights=None,
                 **kwargs):
        """Back-compat: 允許舊參數名 `grip_traj_topic`（topic）
        若收到 `grip_traj_topic='/xarm_gripper_traj_controller/joint_trajectory'`，
        會自動轉成 `grip_action_name='/xarm_gripper_traj_controller/follow_joint_trajectory'`。
        """
        # 舊參數名相容處理
        compat_topic = kwargs.pop("grip_traj_topic", None)
        if isinstance(compat_topic, str) and len(compat_topic) > 0:
            if compat_topic.endswith("/joint_trajectory"):
                ns = compat_topic.rsplit("/", 1)[0]
                grip_action_name = f"{ns}/follow_joint_trajectory"
            else:
                # 若使用者直接給了控制器 namespace，也直接補 action 名
                grip_action_name = f"{compat_topic}/follow_joint_trajectory"

        super().__init__()
        self.max_steps = max_steps
        self.vlm_interval = int(vlm_interval)
        self.vlm_prompt = vlm_prompt
        self.camera_topic = camera_topic
        self._last_vlm_score = 0.0
        self._last_vlm_step = -9999
        self.step_count = 0
        self.gripper_joint_name = gripper_joint_name
        # 旋鈕保存
        self.arm_step_rad = float(arm_step_rad)
        self.arm_limit_margin = float(arm_limit_margin)
        self.arm_time_sec = float(arm_time_sec)
        self.grip_min = float(grip_min)
        self.grip_max = float(grip_max)
        self.grip_step = float(grip_step)
        self.grip_time_sec = float(grip_time_sec)
        self.joint_weights = np.asarray(joint_weights if joint_weights is not None else [1,1,1,1,1,1], dtype=np.float32)

        # ROS
        self.node: Node = Node("xarm6_gym_env")
        self.arm_pub = self.node.create_publisher(JointTrajectory, arm_traj_topic, 10)
        self.grip_ac = ActionClient(self.node, FollowJointTrajectory, grip_action_name)
        self.grip_ac.wait_for_server(timeout_sec=2.0)
        self.state_sub = self.node.create_subscription(JointState, "/joint_states", self._joint_state_cb, 10)
        self.image_sub = self.node.create_subscription(Image, self.camera_topic, self._image_cb, 10)

        self.tf_buffer = Buffer(); self.tf_listener = TransformListener(self.tf_buffer, self.node)
        self.bridge = CvBridge(); self.latest_image = None

        # Spaces
        self.action_space = gym.spaces.Box(low=np.array([-1]*7, dtype=np.float32), high=np.array([1]*7, dtype=np.float32))
        self.observation_space = SpaceDict({
            "state": Box(low=-1.0, high=1.0, shape=(7 + 3*2,), dtype=np.float32),
            "image": Box(low=0, high=255, shape=(IMG_H, IMG_W, 3), dtype=np.uint8),
        })
        self.current_joint_state = np.zeros(6, dtype=np.float32)
        self.gripper_angle: float = self.grip_max  # 初始張開

        # 簡化：兩個物體位置，用於幾何與等距
        self.objects = [np.array([0.5, 0.0, 0.05], np.float32), np.array([0.6, 0.0, 0.05], np.float32)]

    # === ROS Callbacks ===
    def _image_cb(self, msg: Image):
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception:
            self.latest_image = None

    def _joint_state_cb(self, msg: JointState):
        # 依名稱重排到 [joint1..joint6]，避免 JointState 順序亂掉
        try:
            order = [msg.name.index(f"joint{i+1}") for i in range(6)]
            pos = [msg.position[i] for i in order]
            self.current_joint_state = np.array(pos, dtype=np.float32)
        except Exception:
            self.current_joint_state = np.array(msg.position[:6], dtype=np.float32)

    # === Helpers ===
    def _ee_pos(self, joints: np.ndarray) -> np.ndarray:
        try:
            now = rclpy.time.Time()
            tf: TransformStamped = self.tf_buffer.lookup_transform("base_link", "ee_link", now, timeout=rclpy.duration.Duration(seconds=0.1))
            p = tf.transform.translation
            return np.array([p.x, p.y, p.z], dtype=np.float32)
        except Exception:
            x = 0.3 + 0.2*np.cos(joints[0]) + 0.2*np.cos(joints[0] + joints[1])
            y = 0.0 + 0.2*np.sin(joints[0]) + 0.2*np.sin(joints[0] + joints[1])
            z = 0.05 + 0.1*(joints[2] + 1.5)
            return np.array([x, y, max(0.0, z)], dtype=np.float32)

    def _soft_clip_workspace(self, p: np.ndarray) -> np.ndarray:
        return np.array([
            np.clip(p[0], *WORKSPACE["x"]),
            np.clip(p[1], *WORKSPACE["y"]),
            np.clip(p[2], *WORKSPACE["z"]),
        ], np.float32)

    def _normalize_obs(self, raw_state: np.ndarray) -> np.ndarray:
        joint_scaled = 2*(self.current_joint_state - self._LIMITS[:,0])/(self._LIMITS[:,1]-self._LIMITS[:,0]) - 1
        grip_scaled  = 2*(self.gripper_angle - self.grip_min)/(self.grip_max - self.grip_min) - 1
        return np.concatenate([joint_scaled, [grip_scaled], raw_state[7:]], dtype=np.float32)

    def _layout_score(self):
        P = np.stack(self.objects, axis=0)
        y_std = np.std(P[:,1])
        xs = np.sort(P[:,0]); gaps = np.diff(xs)
        gap_var = np.var(gaps) if len(gaps)>0 else 0.0
        return 0.5*np.exp(-5.0*y_std) + 0.5*np.exp(-10.0*gap_var)

    def _reward(self, obs: dict) -> float:
        ee = self._ee_pos(self.current_joint_state)
        dists = np.linalg.norm(np.stack(self.objects) - ee, axis=1)
        r_geom = max(0.0, 1.0 - dists.min()/0.5)
        # 越界懲罰
        for p in self.objects:
            x_ok = WORKSPACE["x"][0] <= p[0] <= WORKSPACE["x"][1]
            y_ok = WORKSPACE["y"][0] <= p[1] <= WORKSPACE["y"][1]
            z_ok = p[2] >= 0.0
            if not (x_ok and y_ok and z_ok):
                r_geom -= 0.5
        r_layout = self._layout_score()
        if self.vlm_interval > 0 and (self.step_count - self._last_vlm_step) >= self.vlm_interval:
            if self.latest_image is not None:
                img = cv2.resize(self.latest_image, (IMG_W, IMG_H))
                self._last_vlm_score = float(get_vlm_score(img, prompt=self.vlm_prompt))
            self._last_vlm_step = self.step_count
        r_vlm = self._last_vlm_score
        return 0.7*(0.6*r_geom + 0.4*r_layout) + 0.3*r_vlm

    def _obs(self):
        raw = list(self.current_joint_state) + [self.gripper_angle]
        for p in self.objects: raw.extend(p)
        img = np.zeros((IMG_H, IMG_W, 3), np.uint8) if self.latest_image is None else cv2.resize(self.latest_image, (IMG_W, IMG_H))
        return {"state": self._normalize_obs(np.asarray(raw, np.float32)), "image": img}

    # === Gripper via Action ===
    def _send_gripper_goal(self, pos: float):
        pos = float(np.clip(pos, self.grip_min, self.grip_max))
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = [self.gripper_joint_name]
        pt = JointTrajectoryPoint()
        pt.positions = [pos]
        # 以秒轉成 sec/nsec
        sec = int(self.grip_time_sec); nsec = int((self.grip_time_sec - sec) * 1e9)
        pt.time_from_start.sec = sec
        pt.time_from_start.nanosec = nsec
        goal.trajectory.points = [pt]
        try: self.grip_ac.send_goal_async(goal)
        except Exception: pass

    # === Gym API ===
    def step(self, action: np.ndarray):
        self.step_count += 1
        a = np.clip(action, -1.0, 1.0)
        # 依權重縮放，步長更可控
        a_scaled = a[:6] * self.joint_weights
        target = self.current_joint_state + self.arm_step_rad * a_scaled
        # 安全邊界夾限
        lo = self._LIMITS[:,0] + self.arm_limit_margin
        hi = self._LIMITS[:,1] - self.arm_limit_margin
        target = np.clip(target, lo, hi)

        # 發 arm 軌跡（topic）
        arm = JointTrajectory(); arm.joint_names = [f"joint{i+1}" for i in range(6)]
        pt = JointTrajectoryPoint(); pt.positions = target.tolist()
        sec = int(self.arm_time_sec); nsec = int((self.arm_time_sec - sec) * 1e9)
        pt.time_from_start.sec = sec; pt.time_from_start.nanosec = nsec
        arm.points.append(pt); self.arm_pub.publish(arm)

        # gripper 走 action（非阻塞）
        self.gripper_angle = float(np.clip(self.gripper_angle + self.grip_step * a[6], self.grip_min, self.grip_max))
        self._send_gripper_goal(self.gripper_angle)

        rclpy.spin_once(self.node, timeout_sec=0.05)

        # 簡化的「抓取後擺正」：當 gripper>0.6，將最近物體 y 對齊 0，模擬整理
        ee = self._ee_pos(self.current_joint_state)
        if self.gripper_angle > (self.grip_min + 0.7*(self.grip_max - self.grip_min)):
            i = int(np.argmin(np.linalg.norm(np.stack(self.objects) - ee, axis=1)))
            self.objects[i] = self._soft_clip_workspace(np.array([ee[0], 0.0, 0.05], np.float32))

        obs = self._obs(); r = self._reward(obs)
        terminated = False
        truncated = self.step_count >= self.max_steps
        info = {"gripper_position": self.gripper_angle}
        return obs, r, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.current_joint_state[:] = 0.0
        self.gripper_angle = self.grip_max
        self._send_gripper_goal(self.gripper_angle)
        self.objects = [np.array([0.5, 0.0, 0.05], np.float32), np.array([0.6, 0.0, 0.05], np.float32)]
        return self._obs(), {}

    def close(self):
        if rclpy.ok():
            self.node.destroy_node(); rclpy.shutdown()