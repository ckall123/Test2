from rclpy.action import ActionClient
from rclpy.qos import qos_profile_sensor_data
from rclpy.node import Node
from rclpy.executors import Executor
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint
from sensor_msgs.msg import JointState
import math
from typing import List, Dict
import numpy as np
import threading

ARM_JOINT_NAMES = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
GRIPPER_JOINT_NAME = 'drive_joint'
GRIPPER_MIN = 0.0  # 請根據實際開合範圍調整
GRIPPER_MAX = 0.8552
JOINT_LIMITS = [
    (-math.radians(178), math.radians(178)),
    (-math.radians(118), math.radians(120)),
    (-math.radians(178), math.radians(11)),
    (-math.radians(178), math.radians(178)),
    (-math.radians(97),  math.radians(178)),
    (-math.radians(178), math.radians(178))
]

class MoveItController:
    def __init__(self, node: Node, executor: Executor):
        self.node = node
        self.executor = executor
        self.client = ActionClient(self.node, MoveGroup, '/move_action')

        self._joint_state_lock = threading.Lock()
        self._joint_state_ready = False
        self._joint_state_map: Dict[str, float] = {}

        self._js_sub = self.node.create_subscription(
            JointState,
            '/joint_states',
            self._joint_state_callback,
            qos_profile_sensor_data
        )

    def _joint_state_callback(self, msg: JointState):
        with self._joint_state_lock:
            for name, pos in zip(msg.name, msg.position):
                self._joint_state_map[name] = pos
            self._joint_state_ready = True

    def _wait_joint_state(self, timeout_sec=2.0):
        import time
        start = time.time()
        while not self._joint_state_ready and time.time() - start < timeout_sec:
            rclpy.spin_once(self.node, timeout_sec=0.1)
        if not self._joint_state_ready:
            raise RuntimeError("❌ 無法獲取 joint_states，請確認控制器正常發布")

    def get_joint_positions(self) -> np.ndarray:
        self._wait_joint_state()
        with self._joint_state_lock:
            try:
                return np.array([self._joint_state_map[name] for name in ARM_JOINT_NAMES])
            except KeyError as e:
                raise RuntimeError(f"❌ 缺少關節資料: {e}")

    def get_gripper_state(self) -> float:
        self._wait_joint_state()
        with self._joint_state_lock:
            if GRIPPER_JOINT_NAME not in self._joint_state_map:
                raise RuntimeError("❌ 找不到夾爪關節")
            pos = self._joint_state_map[GRIPPER_JOINT_NAME]
            norm = (pos - GRIPPER_MIN) / (GRIPPER_MAX - GRIPPER_MIN)
            return float(np.clip(norm, 0.0, 1.0))

    def move_arm(self, joint_positions: List[float]):
        if len(joint_positions) != len(ARM_JOINT_NAMES):
            self.node.get_logger().error('❌ arm joint 數量錯誤')
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
        return self._send_goal_blocking(goal_msg, 'arm')

    def move_gripper(self, position: float):
        position = min(max(position, GRIPPER_MIN), GRIPPER_MAX)

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
        return self._send_goal_blocking(goal_msg, 'gripper')

    def _send_goal_blocking(self, goal_msg, desc):
        self.client.wait_for_server()
        send_future = self.client.send_goal_async(goal_msg)
        self.executor.spin_until_future_complete(send_future)
        goal_handle = send_future.result()

        if not goal_handle.accepted:
            self.node.get_logger().error(f'❌ {desc} 目標被拒絕')
            return False, f'{desc} goal rejected'

        self.node.get_logger().info(f'✅ {desc} 目標已接受，等待結果...')
        result_future = goal_handle.get_result_async()
        self.executor.spin_until_future_complete(result_future)
        result = result_future.result().result

        if result.error_code.val == 1:
            self.node.get_logger().info(f'🎉 {desc} 成功完成')
            return True, ''
        else:
            self.node.get_logger().error(f'⚠️ {desc} 失敗，錯誤碼：{result.error_code.val}')
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
        home_joints = [0.0] * 6
        return self.plan_and_execute(home_joints, 0.0)


if __name__ == '__main__':
    import rclpy
    from rclpy.executors import SingleThreadedExecutor

    rclpy.init()
    node = rclpy.create_node('test_moveit_controller')
    executor = SingleThreadedExecutor()
    executor.add_node(node)

    controller = MoveItController(node, executor)
    target = [math.radians(50), math.radians(-30), math.radians(-90), math.radians(0), math.radians(45), math.radians(0)]
    success, error = controller.plan_and_execute(target, 0.85)

    if success:
        node.get_logger().info("✨ 控制全部完成")
    else:
        node.get_logger().error(f"控制失敗：{error}")

    node.destroy_node()
    rclpy.shutdown()