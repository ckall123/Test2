# =========================
# FILE: envs/_ros_helpers.py
# =========================
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from rclpy.time import Time

from sensor_msgs.msg import JointState, Image
from rcl_interfaces.msg import Log
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import TransformStamped
import tf2_ros

import numpy as np

try:
    from cv_bridge import CvBridge
    _HAS_BRIDGE = True
except Exception:
    _HAS_BRIDGE = False

try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False


@dataclass
class TfPose:
    xyz: np.ndarray  # (3,)
    quat: np.ndarray # (4,) xyzw


class RosHelpers:
    """將 ROS2 常用 I/O 打包：TF、/joint_states、/rosout、影像、trajectory publisher。"""
    def __init__(self,
                 node_name: str = "xarm6_gym_env",
                 arm_traj_topic: str = "/xarm6_traj_controller/joint_trajectory",
                 grip_traj_topic: str = "/xarm_gripper_traj_controller/joint_trajectory",
                 joint_names: Optional[List[str]] = None,
                 gripper_joint_name: str = "drive_joint",
                 camera_topic: Optional[str] = None,
                 image_size: Tuple[int, int] = (224, 224)):
        if not rclpy.ok():
            rclpy.init()
        self.node: Node = rclpy.create_node(node_name)

        # ---- TF ----
        self.tf_buffer = tf2_ros.Buffer(cache_time=rclpy.duration.Duration(seconds=5.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self.node)

        # ---- Publishers ----
        qos_pub = QoSProfile(depth=1)
        self.arm_pub = self.node.create_publisher(JointTrajectory, arm_traj_topic, qos_pub)
        self.grip_pub = self.node.create_publisher(JointTrajectory, grip_traj_topic, qos_pub)

        # ---- Joint states ----
        self.joint_state: Dict[str, float] = {}
        qos_sub = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.node.create_subscription(JointState, "/joint_states", self._joint_cb, qos_sub)

        # ---- rosout（抓 tolerance error）----
        self.violation_flag = False
        self.node.create_subscription(Log, "/rosout", self._rosout_cb, 10)

        # ---- image ----
        self.camera_topic = camera_topic
        self._image_w, self._image_h = image_size
        self._bridge = CvBridge() if _HAS_BRIDGE else None
        self._last_image = None
        if camera_topic:
            self.node.create_subscription(Image, camera_topic, self._image_cb, qos_sub)

        # ---- Names ----
        self.joint_names = joint_names or [
            "joint1", "joint2", "joint3", "joint4", "joint5", "joint6"
        ]
        self.gripper_joint_name = gripper_joint_name

    # --------- Callbacks ---------
    def _joint_cb(self, msg: JointState):
        for name, pos in zip(msg.name, msg.position):
            self.joint_state[name] = float(pos)

    def _rosout_cb(self, msg: Log):
        txt = (msg.msg or "").lower()
        if "state tolerances failed" in txt or "goal_time_tolerance" in txt or "aborted" in txt:
            self.violation_flag = True

    def _image_cb(self, msg: Image):
        if not self._bridge:
            return
        try:
            cv_img = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            if _HAS_CV2:
                cv_img = cv2.resize(cv_img, (self._image_w, self._image_h), interpolation=cv2.INTER_AREA)
                cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            else:
                cv_img = cv_img[..., ::-1]
            self._last_image = np.asarray(cv_img, dtype=np.uint8)
        except Exception:
            pass

    # --------- Utilities ---------
    def spin_once(self, timeout: float = 0.01):
        rclpy.spin_once(self.node, timeout_sec=timeout)

    def get_joint_positions(self) -> np.ndarray:
        return np.array([self.joint_state.get(n, 0.0) for n in self.joint_names], dtype=np.float32)

    def get_gripper_pos(self) -> float:
        return float(self.joint_state.get(self.gripper_joint_name, 0.0))

    def clear_violation(self):
        self.violation_flag = False

    def get_tf_pose(self, target_frame: str, source_frame: str = "world", timeout: float = 0.2) -> Optional[TfPose]:
        try:
            ts: TransformStamped = self.tf_buffer.lookup_transform(
                source_frame, target_frame, Time(), rclpy.duration.Duration(seconds=timeout)
            )
            t = ts.transform.translation
            q = ts.transform.rotation
            return TfPose(
                xyz=np.array([t.x, t.y, t.z], dtype=np.float32),
                quat=np.array([q.x, q.y, q.z, q.w], dtype=np.float32),
            )
        except Exception:
            return None

    def send_arm_traj(self, target_positions: List[float], duration: float = 0.25):
        msg = JointTrajectory()
        msg.joint_names = list(self.joint_names)
        pt = JointTrajectoryPoint()
        pt.positions = list(map(float, target_positions))
        pt.time_from_start = rclpy.duration.Duration(seconds=max(0.05, float(duration))).to_msg()
        msg.points = [pt]
        self.arm_pub.publish(msg)

    def send_gripper_traj(self, position: float, duration: float = 0.25):
        msg = JointTrajectory()
        msg.joint_names = [self.gripper_joint_name]
        pt = JointTrajectoryPoint()
        pt.positions = [float(position)]
        pt.time_from_start = rclpy.duration.Duration(seconds=max(0.05, float(duration))).to_msg()
        msg.points = [pt]
        self.grip_pub.publish(msg)

    def get_image(self, fill_if_none: bool = True) -> Optional[np.ndarray]:
        if self._last_image is None and fill_if_none:
            return np.zeros((self._image_h, self._image_w, 3), dtype=np.uint8)
        return self._last_image
