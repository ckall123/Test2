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

    def move_arm(self, joint_positions: List[float], callback: Optional[callable] = None):
        if len(joint_positions) != len(ARM_JOINT_NAMES):
            self.get_logger().error('❌ arm joint 數量錯誤')
            return

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
        self._send_goal(goal_msg, callback, desc='arm')

    def move_gripper(self, position: float, callback: Optional[callable] = None):
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
        self._send_goal(goal_msg, callback, desc='gripper')

    def _send_goal(self, goal_msg, callback, desc='move'):
        def goal_response(future):
            goal_handle = future.result()
            if not goal_handle.accepted:
                self.get_logger().error(f'❌ {desc} 目標被拒絕')
                return

            self.get_logger().info(f'✅ {desc} 目標已接受，等待結果...')
            goal_handle.get_result_async().add_done_callback(result_response)

        def result_response(future):
            result = future.result().result
            if result.error_code.val == 1:
                self.get_logger().info(f'🎉 {desc} 成功完成')
                if callback:
                    callback()
            else:
                self.get_logger().error(f'⚠️ {desc} 失敗，錯誤碼：{result.error_code.val}')

        self.client.wait_for_server()
        self.client.send_goal_async(goal_msg).add_done_callback(goal_response)


def main():
    rclpy.init()
    controller = MoveItController()

    def after_grip():
        controller.get_logger().info("✨ 控制全部完成")
        rclpy.shutdown()

    def after_arm():
        controller.move_gripper(0.85, callback=after_grip)

    target = [math.radians(0), math.radians(-30), math.radians(-90), math.radians(0), math.radians(45), math.radians(0)]
    controller.move_arm(target, callback=after_arm)

    rclpy.spin(controller)


if __name__ == '__main__':
    main()