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

from vlm.sync_api import get_vlm_score
from reward.reward import layout_score, geom_reward, final_reward

import cv2
from cv_bridge import CvBridge

from rclpy.duration import Duration
from rclpy.time import Time

from objects.sim_object import SimObject
from rl_utils.gripper_control import attach as srv_attach, detach as srv_detach

IMG_H, IMG_W = 64, 64
WORKSPACE = {"x": (0.30, 0.80), "y": (-0.30, 0.30), "z": (0.00, 0.40)}
N_OBJECTS = 2  # 觀測固定長度；多的截斷、少的補零

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
                 # 抓取服務參數（attach/detach）
                 robot_model="UF_ROBOT", gripper_link="right_finger",
                 **kwargs):
        # 舊參數名相容
        compat_topic = kwargs.pop("grip_traj_topic", None)
        if isinstance(compat_topic, str) and len(compat_topic) > 0:
            if compat_topic.endswith("/joint_trajectory"):
                ns = compat_topic.rsplit("/", 1)[0]
                grip_action_name = f"{ns}/follow_joint_trajectory"
            else:
                grip_action_name = f"{compat_topic}/follow_joint_trajectory"

        super().__init__()
        self.max_steps = max_steps
        self.vlm_interval = int(vlm_interval)
        self.vlm_prompt = vlm_prompt
        self.camera_topic = camera_topic
        self._last_vlm_score = 0.5
        self.step_count = 0
        self._step_since_vlm = 0
        self.gripper_joint_name = gripper_joint_name

        self.robot_model = robot_model
        self.gripper_link = gripper_link
        self.attached_index = None  # 目前被抓的物件索引

        # 旋鈕
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
            "state": Box(low=-1.0, high=1.0, shape=(7 + 3*N_OBJECTS,), dtype=np.float32),
            "image": Box(low=0, high=255, shape=(IMG_H, IMG_W, 3), dtype=np.uint8),
        })
        self.current_joint_state = np.zeros(6, dtype=np.float32)
        self.gripper_angle: float = self.grip_max

        # 兩個物件（可替換成 spawner 產生）
        self.objects = [
            SimObject(name="beer_can_1", position=np.array([0.5, 0.0, 0.05], np.float32)),
            SimObject(name="beer_can_2", position=np.array([0.6, 0.0, 0.05], np.float32)),
        ]

    # === ROS Callbacks ===
    def _image_cb(self, msg: Image):
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception:
            self.latest_image = None

    def _joint_state_cb(self, msg: JointState):
        try:
            order = [msg.name.index(f"joint{i+1}") for i in range(6)]
            pos = [msg.position[i] for i in order]
            self.current_joint_state = np.array(pos, dtype=np.float32)
        except Exception:
            self.current_joint_state = np.array(msg.position[:6], dtype=np.float32)

    # === Helpers ===
    def _ee_pos(self, joints: np.ndarray) -> np.ndarray:
        try:
            tf = self.tf_buffer.lookup_transform(
                "base_link", "ee_link", Time(), timeout=Duration(seconds=0.1)
            )
            p = tf.transform.translation
            return np.array([p.x, p.y, p.z], dtype=np.float32)
        except Exception:
            x = 0.3 + 0.2*np.cos(joints[0]) + 0.2*np.cos(joints[0] + joints[1])
            y = 0.0 + 0.2*np.sin(joints[0]) + 0.2*np.sin(joints[0] + joints[1])
            z = 0.05 + 0.1*(joints[2] + 1.5)
            return np.array([x, y, max(0.0, z)], dtype=np.float32)

    def _normalize_obs(self, raw_state: np.ndarray) -> np.ndarray:
        joint_scaled = 2*(self.current_joint_state - self._LIMITS[:,0])/(self._LIMITS[:,1]-self._LIMITS[:,0]) - 1
        grip_scaled  = 2*(self.gripper_angle - self.grip_min)/(self.grip_max - self.grip_min) - 1
        return np.concatenate([joint_scaled, [grip_scaled], raw_state[7:]]).astype(np.float32)

    def _layout_score(self):
        P = [obj.position for obj in self.objects]
        return layout_score(P)

    # === Low-level controls ===
    def _send_gripper_goal(self, pos: float):
        pos = float(np.clip(pos, self.grip_min, self.grip_max))
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = [self.gripper_joint_name]
        pt = JointTrajectoryPoint()
        pt.positions = [pos]
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
        a_scaled = a[:6] * self.joint_weights
        target = self.current_joint_state + self.arm_step_rad * a_scaled
        lo = self._LIMITS[:,0] + self.arm_limit_margin
        hi = self._LIMITS[:,1] - self.arm_limit_margin
        target = np.clip(target, lo, hi)

        arm = JointTrajectory(); arm.joint_names = [f"joint{i+1}" for i in range(6)]
        pt = JointTrajectoryPoint(); pt.positions = target.tolist()
        sec = int(self.arm_time_sec); nsec = int((self.arm_time_sec - sec) * 1e9)
        pt.time_from_start.sec = sec; pt.time_from_start.nanosec = nsec
        arm.points.append(pt); self.arm_pub.publish(arm)

        # gripper
        self.gripper_angle = float(np.clip(self.gripper_angle + self.grip_step * a[6], self.grip_min, self.grip_max))
        self._send_gripper_goal(self.gripper_angle)

        rclpy.spin_once(self.node, timeout_sec=0.05)

        # === 簡化的抓取/放下機制（使用 attach/detach 服務） ===
        ee = self._ee_pos(self.current_joint_state)
        dists = np.linalg.norm(np.stack([o.position for o in self.objects]) - ee, axis=1)
        near_idx = int(np.argmin(dists))
        near_dist = float(dists[near_idx])

        close_threshold = 0.08
        grab_threshold = self.grip_min + 0.7*(self.grip_max - self.grip_min)
        release_threshold = self.grip_min + 0.3*(self.grip_max - self.grip_min)

        # 抓取：夾爪關閉且接近，且尚未附著
        if (self.gripper_angle > grab_threshold) and (near_dist < close_threshold):
            if self.attached_index is None:
                try:
                    srv_attach(self.robot_model, self.gripper_link, self.objects[near_idx].name, self.objects[near_idx].link)
                    self.attached_index = near_idx
                    self.objects[near_idx].attached = True
                except Exception:
                    pass
            else:
                # 已抓取，將被抓物體位置對齊到 ee（簡化）
                idx = self.attached_index
                self.objects[idx].position = self._soft_clip_workspace(np.array([ee[0], ee[1], 0.05], np.float32))

        # 放下：夾爪打開
        if (self.gripper_angle < release_threshold) and (self.attached_index is not None):
            idx = self.attached_index
            try:
                srv_detach(self.robot_model, self.gripper_link, self.objects[idx].name, self.objects[idx].link)
            except Exception:
                pass
            self.attached_index = None
            self.objects[idx].attached = False
            # 放下時把 y 對齊 0（模擬整理行為）
            self.objects[idx].position = self._soft_clip_workspace(np.array([ee[0], 0.0, 0.05], np.float32))

        obs = self._obs()

        # === VLM ===
        vlm_score = self._last_vlm_score
        self._step_since_vlm += 1
        need_vlm = (self.vlm_interval == 0) or (self._step_since_vlm >= self.vlm_interval)
        if need_vlm and (self.latest_image is not None):
            try:
                resized = cv2.resize(self.latest_image, (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR)
                vlm_score = float(get_vlm_score(resized, prompt=self.vlm_prompt))
                vlm_score = max(0.0, min(1.0, vlm_score))
                self._last_vlm_score = vlm_score
                self._step_since_vlm = 0
            except Exception:
                pass

        # === Reward ===
        r_geom = geom_reward(ee, [o.position for o in self.objects], WORKSPACE)
        r_layout = self._layout_score()
        reward = final_reward(r_geom, r_layout, self._last_vlm_score)

        terminated = False
        truncated = self.step_count >= self.max_steps
        info = {"gripper_position": self.gripper_angle}
        return obs, float(reward), terminated, truncated, info

    def _soft_clip_workspace(self, p: np.ndarray) -> np.ndarray:
        return np.array([
            np.clip(p[0], *WORKSPACE["x"]),
            np.clip(p[1], *WORKSPACE["y"]),
            np.clip(p[2], *WORKSPACE["z"]),
        ], np.float32)

    def _obs(self):
        raw = list(self.current_joint_state) + [self.gripper_angle]
        for i in range(N_OBJECTS):
            if i < len(self.objects):
                raw.extend(self.objects[i].position.tolist())
            else:
                raw.extend([0.0, 0.0, 0.0])
        img = np.zeros((IMG_H, IMG_W, 3), np.uint8) if self.latest_image is None else cv2.resize(self.latest_image, (IMG_W, IMG_H))
        return {"state": self._normalize_obs(np.asarray(raw, np.float32)), "image": img}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self._step_since_vlm = 0
        self.current_joint_state[:] = 0.0
        self.gripper_angle = self.grip_max
        self._send_gripper_goal(self.gripper_angle)
        self.attached_index = None
        self.objects = [
            SimObject(name="beer_can_1", position=np.array([0.5, 0.0, 0.05], np.float32)),
            SimObject(name="beer_can_2", position=np.array([0.6, 0.0, 0.05], np.float32)),
        ]
        return self._obs(), {}

    def close(self):
        if hasattr(self, "node"):
            try:
                self.node.get_logger().info("Closing env")
            except Exception:
                pass
            try:
                self.node.destroy_node()
            except Exception:
                pass
        # rclpy.shutdown() 由外層負責