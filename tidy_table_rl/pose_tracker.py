import math
import time
import json
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import TransformStamped
from gazebo_msgs.srv import GetEntityState


class BaseExtrinsic:
    """
    Coordinate transform between Gazebo world and robot link_base frame.
    Matches transformation logic in spawner.py.
    """
    def __init__(self,
                 tx=-0.2, ty=-0.500001, tz=1.020995,
                 roll=-0.000003, pitch=0.0, yaw=1.571):
        self.translation = np.array([tx, ty, tz], dtype=float)

        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)

        self.rotation = np.array([
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp,     cp * sr,                cp * cr]
        ], dtype=float)

    def world_to_base(self, pos_world: np.ndarray) -> np.ndarray:
        return self.rotation.T @ (pos_world - self.translation)

    def base_to_world(self, pos_base: np.ndarray) -> np.ndarray:
        return self.rotation @ pos_base + self.translation


class PoseTracker:
    """
    Tracks TCP and object positions in world coordinates using TF and Gazebo API.
    """
    def __init__(self, node: Node, base_extrinsic: BaseExtrinsic = None):
        self.node = node
        self.base_ext = base_extrinsic or BaseExtrinsic()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, node)

        self.gazebo_client = node.create_client(GetEntityState, '/gazebo/get_entity_state')
        if not self.gazebo_client.wait_for_service(timeout_sec=3.0):
            self.node.get_logger().error("❌ Failed to connect to /gazebo/get_entity_state")

    def get_object_pose_world(self, name: str) -> np.ndarray | None:
        req = GetEntityState.Request()
        req.name = name

        future = self.gazebo_client.call_async(req)
        rclpy.spin_until_future_complete(self.node, future)

        if future.result() and future.result().success:
            pos = future.result().state.pose.position
            return np.array([pos.x, pos.y, pos.z], dtype=float)

        self.node.get_logger().warn(f"⚠️ Could not get pose of {name}")
        return None

    def _get_tf_position(self, target: str, source: str) -> np.ndarray | None:
        try:
            tf: TransformStamped = self.tf_buffer.lookup_transform(
                target_frame=target,
                source_frame=source,
                time=rclpy.time.Time(),
                timeout=Duration(seconds=0.5)
            )
            t = tf.transform.translation
            return np.array([t.x, t.y, t.z], dtype=float)
        except Exception as e:
            self.node.get_logger().warn(f"❌ TF lookup failed {target}←{source}: {e}")
            return None

    def get_tcp_position_world(self, tcp_frame: str = "link_tcp") -> np.ndarray | None:
        tcp_in_base = self._get_tf_position("link_base", tcp_frame)
        if tcp_in_base is not None:
            return self.base_ext.base_to_world(tcp_in_base)
        return None

    def get_distances(self, object_names: list[str], tcp_frame: str = "link_tcp") -> dict:
        distances = {}
        tcp_world = self.get_tcp_position_world(tcp_frame)

        if tcp_world is None:
            return {name: float("inf") for name in object_names}

        print(f"🔹 TCP (world): x={tcp_world[0]:.3f}, y={tcp_world[1]:.3f}, z={tcp_world[2]:.3f}")

        for name in object_names:
            obj_world = self.get_object_pose_world(name)
            if obj_world is None:
                distances[name] = float("inf")
                continue

            print(f"🔸 {name} (world): x={obj_world[0]:.3f}, y={obj_world[1]:.3f}, z={obj_world[2]:.3f}")
            d = float(np.linalg.norm(tcp_world - obj_world))
            distances[name] = d
            print(f"📏 Distance to {name}: {d:.4f} m\n")

        return distances


def main():
    rclpy.init()
    node = rclpy.create_node("pose_tracker_test")
    tracker = PoseTracker(node)

    with open("config.json", 'r') as f:
        config = json.load(f)

    target_objects = config.get("target_objects", [])

    try:
        while rclpy.ok():
            tracker.get_distances(target_objects)
            print("🔁 Waiting for next update...\n")
            for _ in range(50):
                rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
