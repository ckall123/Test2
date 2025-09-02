"""
ROS helpers (v4)
- Single rclpy.Node hosting:
  • TF2 Buffer/Listener (spin_thread=True)
  • /joint_states subscriber (cached)
  • camera Image subscriber (auto-detect topic; cached)
- Public helpers:
  • get_link_pose(parent, child, rpy=True) -> np.ndarray | None
  • get_gripper_position(joint_name='drive_joint', default=0.0) -> float
  • get_joint_positions(names) -> np.ndarray | None
  • get_image(timeout=0.5, fill_if_none=False) -> np.ndarray[H,W,3] | None
  • attach()/detach() via gazebo link attacher services (optional)

Notes:
- Assumes rclpy.init() was called before constructing RosHelpers.
- All methods are defensive: return None/False/defaults instead of raising.
- Camera topic selection order: env var CAMERA_TOPIC > cfg.camera_topic > auto-detect
  among ['/top_camera/image_raw', '/camera/image_raw'] or any sensor_msgs/msg/Image topic.
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time

from tf2_ros import Buffer, TransformListener
from sensor_msgs.msg import JointState, Image

# Link attacher messages (optional)
try:
    from linkattacher_msgs.srv import AttachLink, DetachLink  # type: ignore
    _HAS_LINK_ATTACHER = True
except Exception:
    _HAS_LINK_ATTACHER = False


@dataclass
class RosConfig:
    attach_srv: str = '/ATTACHLINK'
    detach_srv: str = '/DETACHLINK'
    tf_cache_sec: float = 10.0
    camera_topic: str = ''  # if empty, will auto-detect


class RosHelpers(Node):
    def __init__(self, cfg: Optional[RosConfig] = None):
        super().__init__('xarm6_rl_helpers_client')
        self.cfg = cfg or RosConfig()

        # --- TF2 ---
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)

        # --- JointStates subscriber (cache latest) ---
        self._joint_lock = threading.Lock()
        self._joint_state: Optional[JointState] = None
        self._joint_pos_map: dict[str, float] = {}
        self._sub_js = self.create_subscription(JointState, '/joint_states', self._on_joint_state, 10)

        # --- Image subscriber (lazy) ---
        env_cam = os.environ.get('CAMERA_TOPIC', '').strip()
        self._camera_topic = env_cam or (self.cfg.camera_topic or '')
        self._img_lock = threading.Lock()
        self._img_msg: Optional[Image] = None
        self._img_np: Optional[np.ndarray] = None
        self._img_hwh: Optional[Tuple[int, int, int]] = None  # (H,W,C)
        self._sub_img = None  # type: ignore

        # --- Link attacher clients (optional) ---
        self._has_attacher = False
        if _HAS_LINK_ATTACHER:
            try:
                self._cli_attach = self.create_client(AttachLink, self.cfg.attach_srv)
                self._cli_detach = self.create_client(DetachLink, self.cfg.detach_srv)
                self._has_attacher = True
            except Exception:
                self._has_attacher = False

    # ------------------------------------------------------------------
    # Joint states
    # ------------------------------------------------------------------
    def _on_joint_state(self, msg: JointState) -> None:
        with self._joint_lock:
            self._joint_state = msg
            try:
                self._joint_pos_map = {n: float(p) for n, p in zip(msg.name, msg.position)}
            except Exception:
                pass

    def wait_for_joint_states(self, timeout: float = 1.5) -> bool:
        deadline = self.get_clock().now() + Duration(seconds=float(timeout))
        while rclpy.ok() and self.get_clock().now() < deadline:
            if self._joint_state is not None:
                return True
            rclpy.spin_once(self, timeout_sec=0.05)
        return self._joint_state is not None

    def get_joint_positions(self, names: List[str]) -> Optional[np.ndarray]:
        with self._joint_lock:
            if not self._joint_pos_map:
                return None
            vals = []
            for name in names:
                v = self._joint_pos_map.get(name)
                if v is None:
                    return None
                vals.append(float(v))
            return np.array(vals, dtype=np.float32)

    def get_gripper_position(self, joint_name: str = 'drive_joint', default: float = 0.0) -> float:
        with self._joint_lock:
            if self._joint_pos_map and joint_name in self._joint_pos_map:
                return float(self._joint_pos_map[joint_name])
        # Try once to spin for fresh js
        self.wait_for_joint_states(timeout=0.2)
        with self._joint_lock:
            return float(self._joint_pos_map.get(joint_name, default)) if self._joint_pos_map else float(default)

    # ------------------------------------------------------------------
    # TF / poses
    # ------------------------------------------------------------------
    def get_link_pose(self, parent: str = 'link_base', child: str = 'link_tcp', *, rpy: bool = True, timeout: float = 0.5):
        """Lookup transform parent→child from TF and return numpy array.
        rpy=True → [x,y,z, roll,pitch,yaw] (len=6)
        rpy=False → [x,y,z, qx,qy,qz,qw] (len=7)
        Returns None if not available within timeout.
        """
        deadline = self.get_clock().now() + Duration(seconds=float(timeout))
        tf = None
        while rclpy.ok() and self.get_clock().now() < deadline:
            try:
                tf = self.tf_buffer.lookup_transform(parent, child, Time())
                break
            except Exception:
                rclpy.spin_once(self, timeout_sec=0.02)
        if tf is None:
            return None

        t = tf.transform.translation
        q = tf.transform.rotation
        if rpy:
            r, p, y = _quat_to_euler(q.x, q.y, q.z, q.w)
            return np.array([t.x, t.y, t.z, r, p, y], dtype=np.float32)
        else:
            return np.array([t.x, t.y, t.z, q.x, q.y, q.z, q.w], dtype=np.float32)

    # ------------------------------------------------------------------
    # Camera image
    # ------------------------------------------------------------------
    def _ensure_image_sub(self) -> None:
        if self._sub_img is not None:
            return
        topic = self._select_camera_topic()
        self._camera_topic = topic
        try:
            self._sub_img = self.create_subscription(Image, topic, self._on_image, 10)
            self.get_logger().info(f"Subscribed to camera topic: {topic}")
        except Exception as e:
            self.get_logger().warn(f"Failed to subscribe camera topic '{topic}': {e}")
            self._sub_img = None

    def _select_camera_topic(self) -> str:
        # Priority: env var / cfg provided
        if self._camera_topic:
            return self._camera_topic
        # Prefer known names if available
        preferred = ['/top_camera/image_raw', '/camera/image_raw']
        topics = dict(self.get_topic_names_and_types())
        for cand in preferred:
            tys = topics.get(cand, [])
            if any('sensor_msgs/msg/Image' in t for t in tys):
                return cand
        # Otherwise, pick any Image topic
        for name, tys in topics.items():
            if any('sensor_msgs/msg/Image' in t for t in tys):
                return name
        # Fallback (may or may not exist)
        return '/top_camera/image_raw'

    def _on_image(self, msg: Image) -> None:
        with self._img_lock:
            self._img_msg = msg
            try:
                arr = _rosimg_to_numpy_rgb(msg)
                self._img_np = arr
                self._img_hwh = (arr.shape[0], arr.shape[1], arr.shape[2])
            except Exception:
                # keep last good _img_np
                pass

    def get_image(self, timeout: float = 0.5, fill_if_none: bool = False) -> Optional[np.ndarray]:
        """Return last RGB image as np.uint8 [H,W,3]. If none yet, wait up to timeout.
        If still none and fill_if_none=True, return zero image (640x480x3 default)."""
        self._ensure_image_sub()
        deadline = self.get_clock().now() + Duration(seconds=float(timeout))
        while rclpy.ok() and self._img_np is None and self.get_clock().now() < deadline:
            rclpy.spin_once(self, timeout_sec=0.02)
        with self._img_lock:
            if self._img_np is not None:
                return self._img_np.copy()
        if fill_if_none:
            h, w = 480, 640
            if self._img_hwh is not None:
                h, w = self._img_hwh[0], self._img_hwh[1]
            return np.zeros((h, w, 3), dtype=np.uint8)
        return None

    # ------------------------------------------------------------------
    # Gazebo link attacher (optional)
    # ------------------------------------------------------------------
    def wait_until_ready(self, timeout_sec: float = 3.0) -> None:
        if self._has_attacher:
            self._cli_attach.wait_for_service(timeout_sec=timeout_sec)
            self._cli_detach.wait_for_service(timeout_sec=timeout_sec)

    def attach(self, model1: str, link1: str, model2: str, link2: str, timeout_sec: float = 2.0) -> bool:
        if not self._has_attacher:
            self.get_logger().warn('attach() requested but link attacher services not available')
            return False
        try:
            req = AttachLink.Request()
            req.model1_name = model1
            req.link1_name = link1
            req.model2_name = model2
            req.link2_name = link2
            future = self._cli_attach.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_sec)
            return bool(future.result())
        except Exception as e:
            self.get_logger().warn(f'attach() failed: {e}')
            return False

    def detach(self, model1: str, link1: str, model2: str, link2: str, timeout_sec: float = 2.0) -> bool:
        if not self._has_attacher:
            self.get_logger().warn('detach() requested but link attacher services not available')
            return False
        try:
            req = DetachLink.Request()
            req.model1_name = model1
            req.link1_name = link1
            req.model2_name = model2
            req.link2_name = link2
            future = self._cli_detach.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_sec)
            return bool(future.result())
        except Exception as e:
            self.get_logger().warn(f'detach() failed: {e}')
            return False


# ----------------------------------------------------------------------
# Math & conversion helpers
# ----------------------------------------------------------------------

def _quat_to_euler(x: float, y: float, z: float, w: float) -> Tuple[float, float, float]:
    # roll
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    # pitch
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.pi / 2 * np.sign(sinp)
    else:
        pitch = np.arcsin(sinp)
    # yaw
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return float(roll), float(pitch), float(yaw)


def _rosimg_to_numpy_rgb(msg: Image) -> np.ndarray:
    h, w = int(msg.height), int(msg.width)
    enc = (msg.encoding or '').lower()
    buf = np.frombuffer(msg.data, dtype=np.uint8)

    if enc in ('rgb8', 'rgb_8'):
        arr = buf.reshape(h, w, 3)
        return arr
    if enc in ('bgr8', 'bgr_8'):
        arr = buf.reshape(h, w, 3)
        return arr[..., ::-1]  # BGR -> RGB
    if enc in ('rgba8', 'rgba_8'):
        arr = buf.reshape(h, w, 4)
        return arr[..., :3]
    if enc in ('bgra8', 'bgra_8'):
        arr = buf.reshape(h, w, 4)
        arr = arr[..., :3]
        return arr[..., ::-1]  # BGR -> RGB

    # Fallback: try 3 channels
    try:
        arr = buf.reshape(h, w, 3)
        return arr
    except Exception:
        # Last resort: zeros
        return np.zeros((h, w, 3), dtype=np.uint8)


__all__ = [
    'RosHelpers',
    'RosConfig',
]
