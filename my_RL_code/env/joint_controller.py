#!/usr/bin/env python3
"""
Joint Controller 控制模組：
- 控制 XArm6 手臂 6 軸位置
- 控制夾爪張開/關閉程度
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import numpy as np


class JointController(Node):
    def __init__(self, node_name: str = "joint_controller"):
        super().__init__(node_name)

        # Action clients
        self.arm_client = ActionClient(
            self, FollowJointTrajectory, "/xarm6_traj_controller/follow_joint_trajectory"
        )
        self.gripper_client = ActionClient(
            self, FollowJointTrajectory, "/xarm_gripper_traj_controller/follow_joint_trajectory"
        )

        self.arm_joint_names = [f"joint{i}" for i in range(1, 7)]
        self.gripper_joint_name = "drive_joint"

    def _send_trajectory(self, joint_names: list[str], positions: list[float],
                         duration: float, client: ActionClient) -> bool:
        """共用的發送函數（手臂/夾爪皆可）"""
        if not client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error("❌ Action server 未就緒")
            return False

        traj = JointTrajectory()
        traj.joint_names = joint_names
        point = JointTrajectoryPoint()
        point.positions = [float(p) for p in positions]
        point.time_from_start = Duration(
            sec=int(duration), nanosec=int((duration % 1) * 1e9)
        )
        traj.points.append(point)

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj

        send_future = client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future)
        goal_handle = send_future.result()

        if not goal_handle or not goal_handle.accepted:
            self.get_logger().warn("⚠️ 控制指令被拒絕")
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        return True

    def move_arm(self, joint_positions: list[float], duration: float = 1.0) -> bool:
        """移動機械手臂到指定角度（共6軸）"""
        if len(joint_positions) != 6:
            raise ValueError("❗ 需要提供 6 個關節角度")
        return self._send_trajectory(
            self.arm_joint_names, joint_positions, duration, self.arm_client
        )

    def control_gripper(self, open_ratio: float, duration: float = 0.5) -> bool:
        """
        控制夾爪開合（0.0=關、1.0=開）
        """
        MIN_POS, MAX_POS = 0.0, 0.8  # ← 依你模型/實體微調
        target = np.clip(open_ratio, 0.0, 1.0) * (MAX_POS - MIN_POS) + MIN_POS
        return self._send_trajectory(
            [self.gripper_joint_name], [target], duration, self.gripper_client
        )


def main():
    rclpy.init()
    node = JointController()

    # 測試手臂（歸零）
    node.get_logger().info("🔧 測試移動手臂到初始位置...")
    node.move_arm([0, 0, 0, 0, 0, 0], duration=1.5)

    # 測試夾爪（開合）
    node.get_logger().info("🤖 測試開合夾爪...")
    node.control_gripper(1.0)
    node.control_gripper(0.0)

    rclpy.shutdown()


if __name__ == "__main__":
    main()
