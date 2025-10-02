#!/usr/bin/env python3
"""
Gripper 角度追蹤器
- 訂閱 `/joint_states`
- 擷取指定夾爪關節的角度（通常是 drive_joint）
"""

import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class GripperAngleTracker(Node):
    def __init__(self, joint_name: str = "drive_joint", node_name: str = "gripper_angle_tracker"):
        super().__init__(node_name)
        self.joint_name = joint_name
        self.current_angle = 0.0
        self.last_msg = None

        # 訂閱 joint state
        self.create_subscription(JointState, "/joint_states", self._on_joint_state, 10)

    def _on_joint_state(self, msg: JointState):
        self.last_msg = msg
        try:
            index = msg.name.index(self.joint_name)
            self.current_angle = msg.position[index]
        except ValueError:
            self.get_logger().warn(f"關節 '{self.joint_name}' 不存在於 joint_states 中")

    def get_angle(self) -> float:
        """回傳目前夾爪角度（弧度）"""
        return float(self.current_angle)


def main():
    rclpy.init()
    tracker = GripperAngleTracker()
    print("📡 正在監聽 /joint_states ...")

    try:
        for _ in range(10):
            rclpy.spin_once(tracker, timeout_sec=0.1)
            angle = tracker.get_angle()
            print(f"🔍 目前夾爪角度（rad）: {angle:.4f}")
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass

    rclpy.shutdown()
    print("🛑 測試結束")


if __name__ == "__main__":
    main()
