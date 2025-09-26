#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class JointDebugger(Node):
    def __init__(self):
        super().__init__('joint_debugger')
        self.create_subscription(JointState, '/joint_states', self.joint_callback, 10)
        print("📡 正在監聽 /joint_states 並列出所有關節與角度...")

    def joint_callback(self, msg: JointState):
        print("\n🔍 目前關節狀態：")
        for name, pos in zip(msg.name, msg.position):
            print(f"  {name:>25s} : {pos:.4f}")
        print("🛑 測試結束")
        rclpy.shutdown()


def main():
    rclpy.init()
    node = JointDebugger()
    rclpy.spin(node)


if __name__ == '__main__':
    main()
