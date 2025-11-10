import math
import json
import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import TransformStamped
from gazebo_msgs.srv import GetEntityState


class BaseExtrinsic:
    def __init__(self, tx=-0.2, ty=-0.500001, tz=1.020995, roll=0.0, pitch=0.0, yaw=1.571):
        self.translation = np.array([tx, ty, tz], dtype=float)
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        self.rotation = np.array([
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp,     cp * sr,                cp * cr]
        ])

    def base_to_world(self, pos_base: np.ndarray) -> np.ndarray:
        return self.rotation @ pos_base + self.translation


def _quat_to_yaw(qx, qy, qz, qw):
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny, cosy)


class PoseTracker:
    def __init__(self, node: Node, executor, base_extrinsic: BaseExtrinsic = None):
        self.node = node
        self.executor = executor
        self.base_ext = base_extrinsic or BaseExtrinsic()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, node)
        self.client = node.create_client(GetEntityState, '/gazebo/get_entity_state')
        self.client.wait_for_service(timeout_sec=3.0)

    def _get_tf_position(self, target: str, source: str) -> TransformStamped | None:
        try:
            return self.tf_buffer.lookup_transform(
                target_frame=target,
                source_frame=source,
                time=rclpy.time.Time(),
                timeout=Duration(seconds=0.5)
            )
        except Exception:
            return None

    def get_tcp_pose(self, tcp_frame: str = "link_tcp") -> tuple[np.ndarray, float] | None:
        self.executor.spin_once(timeout_sec=0.1)
        tf = self._get_tf_position("link_base", tcp_frame)
        if tf is None:
            return None
        t = tf.transform.translation
        pos_world = self.base_ext.base_to_world(np.array([t.x, t.y, t.z]))
        q = tf.transform.rotation
        yaw_base = _quat_to_yaw(q.x, q.y, q.z, q.w)
        base_yaw = math.atan2(self.base_ext.rotation[1, 0], self.base_ext.rotation[0, 0])
        yaw = yaw_base + base_yaw
        return pos_world, (yaw + math.pi) % (2 * math.pi) - math.pi

    def get_object_pose_world(self, name: str) -> np.ndarray | None:
        req = GetEntityState.Request()
        req.name = name
        future = self.client.call_async(req)
        self.executor.spin_until_future_complete(future)
        try:
            res = future.result()
        except Exception:
            res = None
        if res and res.success:
            p = res.state.pose.position
            return np.array([p.x, p.y, p.z])
        return None

    def get_object_yaw(self, name: str) -> float | None:
        req = GetEntityState.Request()
        req.name = name
        future = self.client.call_async(req)
        self.executor.spin_until_future_complete(future)
        try:
            res = future.result()
        except Exception:
            res = None
        if res and res.success:
            o = res.state.pose.orientation
            return _quat_to_yaw(o.x, o.y, o.z, o.w)
        return None

    def rel_vec_to(self, name: str, tcp_frame: str = "link_tcp") -> np.ndarray:
        tcp_pose = self.get_tcp_pose(tcp_frame)
        obj_pos = self.get_object_pose_world(name)
        return obj_pos - tcp_pose[0] if tcp_pose and obj_pos is not None else np.full(3, np.inf)

    def get_object_states(self, names: list[str], radius_lookup: dict = None) -> list[dict]:
        results = []
        for name in names:
            req = GetEntityState.Request()
            req.name = name
            future = self.client.call_async(req)
            self.executor.spin_until_future_complete(future)
            try:
                res = future.result()
            except Exception:
                res = None
            if res and res.success:
                p = res.state.pose.position
                o = res.state.pose.orientation
                results.append({
                    "name": name,
                    "pos": np.array([p.x, p.y, p.z]),
                    "yaw": _quat_to_yaw(o.x, o.y, o.z, o.w),
                    "radius": radius_lookup.get(name, 0.02) if radius_lookup else 0.02
                })
        return results


def load_target_names(path="config.json") -> list[str]:
    with open(path) as f:
        cfg = json.load(f)
    return cfg.get("target_objects") or [o["name"] for o in cfg.get("objects", [])]


if __name__ == "__main__":
    rclpy.init()
    node = rclpy.create_node("pose_tracker_test")
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    tracker = PoseTracker(node, executor)
    targets = load_target_names()

    for _ in range(50):
        if tracker.get_tcp_pose() is not None:
            break
        time.sleep(0.1)

    while rclpy.ok():
        tcp = tracker.get_tcp_pose()
        if tcp:
            print(f"🔹 TCP: pos={tcp[0]}, yaw={tcp[1]:.3f}")
        for o in tracker.get_object_states(targets):
            print(f"🔸 {o['name']}: pos={o['pos']}, yaw={o['yaw']:.3f}, radius={o['radius']:.2f}")
        time.sleep(0.1)

    node.destroy_node()
    rclpy.shutdown()
