#!/usr/bin/env python3
"""
pose_tracker.py
提供幾何量測介面，供你的獎勵計算與觀測使用：
- get_object_states(): 回傳所有物體的 pos/yaw/radius，給 reward 計算幾何能量用
- get_tcp_pose(): 回傳 TCP 在 world 的位置與朝向
- get_distances(): 回傳 TCP 到每個物體的距離（觀測用）
- get_object_pose_world(), get_object_yaw(), rel_vec_to(): 個別查詢
"""

import math
import json
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import TransformStamped
from gazebo_msgs.srv import GetEntityState


class BaseExtrinsic:
    def __init__(self, tx=-0.2, ty=-0.500001, tz=1.020995, roll=-0.000003, pitch=0.0, yaw=1.571):
        self.translation = np.array([tx, ty, tz], dtype=float)
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        self.rotation = np.array([
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp,     cp * sr,                cp * cr]
        ], dtype=float)

    def base_to_world(self, pos_base: np.ndarray) -> np.ndarray:
        return self.rotation @ pos_base + self.translation


def _quat_to_yaw(qx, qy, qz, qw):
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


class PoseTracker:
    def __init__(self, node: Node, base_extrinsic: BaseExtrinsic = None):
        self.node = node
        self.base_ext = base_extrinsic or BaseExtrinsic()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, node)
        self.gazebo_client = node.create_client(GetEntityState, '/gazebo/get_entity_state')
        self.gazebo_client.wait_for_service(timeout_sec=3.0)

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
        tf = self._get_tf_position("link_base", tcp_frame)
        if not tf:
            return None

        t = tf.transform.translation
        pos_world = self.base_ext.base_to_world(np.array([t.x, t.y, t.z], dtype=float))

        q = tf.transform.rotation
        yaw_base = _quat_to_yaw(q.x, q.y, q.z, q.w)
        base_yaw = math.atan2(self.base_ext.rotation[1, 0], self.base_ext.rotation[0, 0])
        yaw_world = yaw_base + base_yaw

        if yaw_world > math.pi:
            yaw_world -= 2 * math.pi
        elif yaw_world < -math.pi:
            yaw_world += 2 * math.pi

        return pos_world, float(yaw_world)

    def get_object_pose_world(self, name: str) -> np.ndarray | None:
        req = GetEntityState.Request()
        req.name = name
        future = self.gazebo_client.call_async(req)
        rclpy.spin_until_future_complete(self.node, future)
        if future.result() and future.result().success:
            pos = future.result().state.pose.position
            return np.array([pos.x, pos.y, pos.z], dtype=float)
        return None

    def get_object_yaw(self, name: str) -> float | None:
        req = GetEntityState.Request()
        req.name = name
        future = self.gazebo_client.call_async(req)
        rclpy.spin_until_future_complete(self.node, future)
        if future.result() and future.result().success:
            ori = future.result().state.pose.orientation
            return _quat_to_yaw(ori.x, ori.y, ori.z, ori.w)
        return None

    def rel_vec_to(self, name: str, tcp_frame: str = "link_tcp") -> np.ndarray:
        tcp_pose = self.get_tcp_pose(tcp_frame)
        obj_pos = self.get_object_pose_world(name)
        if tcp_pose is None or obj_pos is None:
            return np.array([np.inf, np.inf, np.inf], dtype=float)  # 保持數值型，方便觀測拼接
        return obj_pos - tcp_pose[0]

    def get_object_states(self, names: list[str], radius_lookup: dict = None) -> list[dict]:
        results = []
        for name in names:
            req = GetEntityState.Request()
            req.name = name
            future = self.gazebo_client.call_async(req)
            rclpy.spin_until_future_complete(self.node, future)
            if future.result() and future.result().success:
                pos = future.result().state.pose.position
                ori = future.result().state.pose.orientation
                yaw = _quat_to_yaw(ori.x, ori.y, ori.z, ori.w)
                results.append({
                    "name": name,
                    "pos": np.array([pos.x, pos.y, pos.z], dtype=float),
                    "yaw": yaw,
                    "radius": radius_lookup.get(name, 0.02) if radius_lookup else 0.02
                })
        return results

    def get_distances(self, object_names: list[str], tcp_frame: str = "link_tcp") -> dict:
        distances = {}
        tcp_pose = self.get_tcp_pose(tcp_frame)
        if tcp_pose is None:
            return {name: float("inf") for name in object_names}

        tcp_pos = tcp_pose[0]
        for name in object_names:
            obj_pos = self.get_object_pose_world(name)
            distances[name] = float(np.linalg.norm(tcp_pos - obj_pos)) if obj_pos is not None else float("inf")
        return distances


def load_target_names(config_path="config.json") -> list[str]:
    with open(config_path, "r") as f:
        cfg = json.load(f)
    return cfg.get("target_objects") or [o["name"] for o in cfg.get("objects", [])]


def main():
    rclpy.init()
    node = rclpy.create_node("pose_tracker_test")
    tracker = PoseTracker(node)
    target_names = load_target_names()

    for _ in range(50):
        if tracker.get_tcp_pose() is not None:
            break
        rclpy.spin_once(node, timeout_sec=0.1)

    while rclpy.ok():
        tcp = tracker.get_tcp_pose()
        if tcp:
            print(f"🔹 TCP (world): x={tcp[0][0]:.3f}, y={tcp[0][1]:.3f}, z={tcp[0][2]:.3f}, yaw={tcp[1]:.3f}")

        objs = tracker.get_object_states(target_names)
        for obj in objs:
            print(f"🔸 {obj['name']}: pos={obj['pos']}, yaw={obj['yaw']:.3f}, radius={obj['radius']:.3f}")

        dists = tracker.get_distances(target_names)
        for name, d in dists.items():
            print(f"📏 {name} dist_to_tcp: {d:.4f} m")

        print("\n---\n")
        for _ in range(10):
            rclpy.spin_once(node, timeout_sec=0.1)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()