import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint
import math
from typing import List, Optional

# 預設 joint 名稱與限制
ARM_JOINT_NAMES = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
GRIPPER_JOINT_NAME = 'drive_joint'
JOINT_LIMITS = [
    (-math.radians(178), math.radians(178)),
    (-math.radians(118), math.radians(120)),
    (-math.radians(178), math.radians(11)),
    (-math.radians(178), math.radians(178)),
    (-math.radians(97),  math.radians(178)),
    (-math.radians(178), math.radians(178))
]

class MoveItController(Node):
    def __init__(self):
        super().__init__('moveit_joint_controller')
        self.client = ActionClient(self, MoveGroup, '/move_action')

    def move_arm(self, joint_positions: List[float]):
        if len(joint_positions) != len(ARM_JOINT_NAMES):
            self.get_logger().error('❌ arm joint 數量錯誤')
            raise ValueError("Invalid joint count")

        goal_msg = MoveGroup.Goal()
        goal_msg.request.group_name = 'xarm6'

        constraint = Constraints()
        for i, (name, pos) in enumerate(zip(ARM_JOINT_NAMES, joint_positions)):
            min_limit, max_limit = JOINT_LIMITS[i]
            clipped = min(max(pos, min_limit), max_limit)

            jc = JointConstraint()
            jc.joint_name = name
            jc.position = clipped
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight = 1.0
            constraint.joint_constraints.append(jc)

        goal_msg.request.goal_constraints.append(constraint)
        return self._send_goal_blocking(goal_msg, desc='arm')

    def move_gripper(self, position: float):
        position = min(max(position, 0.0), 0.8552)

        goal_msg = MoveGroup.Goal()
        goal_msg.request.group_name = 'xarm_gripper'

        constraint = Constraints()
        jc = JointConstraint()
        jc.joint_name = GRIPPER_JOINT_NAME
        jc.position = position
        jc.tolerance_above = 0.01
        jc.tolerance_below = 0.01
        jc.weight = 1.0
        constraint.joint_constraints.append(jc)

        goal_msg.request.goal_constraints.append(constraint)
        return self._send_goal_blocking(goal_msg, desc='gripper')

    def _send_goal_blocking(self, goal_msg, desc='move'):
        self.client.wait_for_server()
        send_future = self.client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, send_future)
        goal_handle = send_future.result()

        if not goal_handle.accepted:
            self.get_logger().error(f'❌ {desc} 目標被拒絕')
            return False, f'{desc} goal rejected'

        self.get_logger().info(f'✅ {desc} 目標已接受，等待結果...')
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result().result

        if result.error_code.val == 1:
            self.get_logger().info(f'🎉 {desc} 成功完成')
            return True, ''
        else:
            self.get_logger().error(f'⚠️ {desc} 失敗，錯誤碼：{result.error_code.val}')
            return False, f'{desc} failed with error code {result.error_code.val}'

    def plan_and_execute(self, joint_goal, gripper_cmd):
        success_arm, err_arm = self.move_arm(joint_goal)
        if not success_arm:
            return False, err_arm

        success_gripper, err_gripper = self.move_gripper(gripper_cmd)
        if not success_gripper:
            return False, err_gripper

        return True, ""

    def go_home(self):
        home_joints = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # 根據你的機器人設定調整
        self.plan_and_execute(home_joints, gripper_cmd=0.0)


def main():
    rclpy.init()
    controller = MoveItController()

    target = [math.radians(0), math.radians(-30), math.radians(-90), math.radians(0), math.radians(45), math.radians(0)]
    success, error = controller.plan_and_execute(target, gripper_cmd=0.85)

    if success:
        controller.get_logger().info("✨ 控制全部完成")
    else:
        controller.get_logger().error(f"控制失敗：{error}")

    controller.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()