# tidy_table_rl/control/moveit_controller.py

import math
import threading
import time
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint

from attach_detach import attach_object, detach_object

JOINT_NAMES = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
JOINT_LIMITS = [
    (-math.radians(178), math.radians(178)),
    (-math.radians(118), math.radians(120)),
    (-math.radians(178), math.radians(11)),
    (-math.radians(178), math.radians(178)),
    (-math.radians(97), math.radians(178)),
    (-math.radians(178), math.radians(178))
]


class MoveItController:
    """
    低階執行層，提供 MoveIt2 控制與 gripper attach/detach 的 API。
    不包含任務邏輯，外部自己決定要抓誰、放哪裡。
    """

    def __init__(self, enable_gripper: bool = True):
        rclpy.init()
        self.node = Node('moveit_controller_node')
        self.arm_client = ActionClient(self.node, MoveGroup, '/move_action')
        self.enable_gripper = enable_gripper

        if self.enable_gripper:
            self.gripper_client = ActionClient(self.node, MoveGroup, '/move_action')

        self._arm_lock = threading.Event()
        self._gripper_lock = threading.Event()
        self.eef_link = 'link6'  # 手臂末端 link 名稱，可改成實際 eef_link

    def shutdown(self):
        """關閉 ROS2 節點"""
        self.node.destroy_node()
        rclpy.shutdown()

    # ------------------------ 手臂控制 ------------------------

    def move_arm(self, joint_positions):
        """移動手臂到指定 6 關節角度"""
        if len(joint_positions) != 6:
            self.node.get_logger().error("❌ 需要 6 個關節位置")
            return False

        for i, angle in enumerate(joint_positions):
            min_limit, max_limit = JOINT_LIMITS[i]
            if angle < min_limit or angle > max_limit:
                self.node.get_logger().error(f'❌ {JOINT_NAMES[i]} 超出範圍')
                return False

        if not self.arm_client.wait_for_server(timeout_sec=3.0):
            self.node.get_logger().error("找不到手臂 MoveIt server")
            return False

        goal_msg = MoveGroup.Goal()
        goal_msg.request.group_name = 'xarm6'

        constraint = Constraints()
        for name, angle in zip(JOINT_NAMES, joint_positions):
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = angle
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight = 1.0
            constraint.joint_constraints.append(jc)

        goal_msg.request.goal_constraints.append(constraint)

        self._arm_lock.clear()
        self.arm_client.send_goal_async(goal_msg).add_done_callback(self._arm_result_callback)

        while not self._arm_lock.is_set():
            rclpy.spin_once(self.node, timeout_sec=0.1)

        return True

    def _arm_result_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.node.get_logger().error('❌ 手臂目標被拒絕')
            self._arm_lock.set()
            return

        self.node.get_logger().info('✅ 手臂目標已接受，等待結果...')
        goal_handle.get_result_async().add_done_callback(self._arm_done_callback)

    def _arm_done_callback(self, future):
        result = future.result().result
        if result.error_code.val == 1:
            self.node.get_logger().info('🎉 手臂移動完成')
        else:
            self.node.get_logger().error(f'⚠️ 手臂移動失敗：{result.error_code.val}')
        self._arm_lock.set()

    # ------------------------ 夾爪控制 ------------------------

    def set_gripper(self, position: float):
        """設定夾爪開合，範圍 0.0 ~ 0.8552"""
        if not self.enable_gripper:
            self.node.get_logger().warn("⚠️ Gripper 控制未啟用，跳過 set_gripper() 呼叫")
            return False

        position = max(0.0, min(position, 0.8552))

        if not self.gripper_client.wait_for_server(timeout_sec=3.0):
            self.node.get_logger().error("找不到夾爪 MoveIt server")
            return False

        goal_msg = MoveGroup.Goal()
        goal_msg.request.group_name = 'xarm_gripper'

        constraint = Constraints()
        jc = JointConstraint()
        jc.joint_name = 'drive_joint'
        jc.position = position
        jc.tolerance_above = 0.01
        jc.tolerance_below = 0.01
        jc.weight = 1.0
        constraint.joint_constraints.append(jc)
        goal_msg.request.goal_constraints.append(constraint)

        self._gripper_lock.clear()
        self.gripper_client.send_goal_async(goal_msg).add_done_callback(self._gripper_result_callback)

        while not self._gripper_lock.is_set():
            rclpy.spin_once(self.node, timeout_sec=0.1)

        return True

    def _gripper_result_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.node.get_logger().error('❌ 夾爪動作被拒絕')
            self._gripper_lock.set()
            return

        self.node.get_logger().info('✅ 夾爪目標已接受，等待結果...')
        goal_handle.get_result_async().add_done_callback(self._gripper_done_callback)

    def _gripper_done_callback(self, future):
        result = future.result().result
        if result.error_code.val == 1:
            self.node.get_logger().info('🎉 夾爪動作完成')
        else:
            self.node.get_logger().error(f'⚠️ 夾爪動作失敗：{result.error_code.val}')
        self._gripper_lock.set()

    # ------------------------ 物件 attach/detach ------------------------

    def grasp(self, object_name: str, gripper_close: float):
        """
        嘗試抓取物件（外部先決定好要夾誰、夾多少）
        """
        self.node.get_logger().info(f"準備抓取 {object_name}...")
        self.set_gripper(0.0)
        time.sleep(0.5)
        self.set_gripper(gripper_close)
        time.sleep(0.5)

        # 直接 attach（可以改成檢查 contact 再 attach）
        attach_object(object_name)
        self.node.get_logger().info(f"🎯 已 attach {object_name}")

        return True

    def release(self, object_name: str):
        """放下物件"""
        self.node.get_logger().info(f"準備放置 {object_name}...")
        detach_object(object_name)
        self.set_gripper(0.0)
        return True


def main():
    controller = MoveItController(enable_gripper=True)

    try:
        print("🎯 移動到初始位置...")
        controller.move_arm([
            math.radians(-176),
            math.radians(-21),
            math.radians(-4),
            math.radians(0),
            math.radians(25),
            math.radians(-176)
        ])
        time.sleep(1)

        # === 🟫 wood_cube_5cm ===
        print("🟫 嘗試夾取 wood_cube_5cm")
        controller.set_gripper(0.0)
        time.sleep(0.5)
        controller.set_gripper(0.45)  # ✅ 依你手爪模型調整
        time.sleep(0.5)
        attach_object("wood_cube_5cm")

        controller.move_arm([
            math.radians(0),
            math.radians(-30),
            math.radians(-90),
            math.radians(0),
            math.radians(0),
            math.radians(0)
        ])
        time.sleep(1)

        print("🟫 放下 wood_cube_5cm")
        detach_object("wood_cube_5cm")
        controller.set_gripper(0.0)

        # === 回到初始點 ===
        controller.move_arm([
            math.radians(-176),
            math.radians(-21),
            math.radians(-4),
            math.radians(0),
            math.radians(25),
            math.radians(-176)
        ])
        time.sleep(1)

        # === 🔵 blue_box ===
        print("🔵 嘗試夾取 blue_box")
        controller.set_gripper(0.0)
        time.sleep(0.5)
        controller.set_gripper(0.52)  # ✅ 依你手爪模型調整
        time.sleep(0.5)
        attach_object("blue_box")

        controller.move_arm([
            math.radians(20),
            math.radians(-20),
            math.radians(-90),
            math.radians(0),
            math.radians(0),
            math.radians(0)
        ])
        time.sleep(1)

        print("🔵 放下 blue_box")
        detach_object("blue_box")
        controller.set_gripper(0.0)

    finally:
        print("🧹 關閉 controller...")
        time.sleep(1)
        controller.shutdown()


if __name__ == '__main__':
    main()

