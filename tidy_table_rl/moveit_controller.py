import math
import threading
from typing import List, Dict

import numpy as np
from sensor_msgs.msg import JointState
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.executors import Executor
from rclpy.qos import qos_profile_sensor_data


ARM_JOINT_NAMES = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
GRIPPER_JOINT_NAME = 'drive_joint'
GRIPPER_MIN = 0.0
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
    """負責控制機械臂與夾爪的 MoveIt client。"""
    def __init__(self, node: Node, executor: Executor):
        self.node = node
        self.executor = executor
        self.client = ActionClient(node, MoveGroup, '/move_action')

        self._joint_state_lock = threading.Lock()
        self._joint_state_map: Dict[str, float] = {}
        self._joint_state_ready = False

        node.create_subscription(
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
            return np.array([self._joint_state_map[name] for name in ARM_JOINT_NAMES])

    def get_gripper_state(self) -> float:
        self._wait_joint_state()
        with self._joint_state_lock:
            pos = self._joint_state_map.get(GRIPPER_JOINT_NAME, GRIPPER_MIN)
            norm = (pos - GRIPPER_MIN) / (GRIPPER_MAX - GRIPPER_MIN)
            return float(np.clip(norm, 0.0, 1.0))

    def move_arm(self, joint_positions: List[float]) -> tuple[bool, str]:
        if len(joint_positions) != len(ARM_JOINT_NAMES):
            return False, "❌ arm joint 數量錯誤"

        goal = MoveGroup.Goal()
        goal.request.group_name = 'xarm6'

        constraint = Constraints()
        for i, (name, pos) in enumerate(zip(ARM_JOINT_NAMES, joint_positions)):
            min_limit, max_limit = JOINT_LIMITS[i]
            pos = np.clip(pos, min_limit, max_limit)

            jc = JointConstraint()
            jc.joint_name = name
            jc.position = pos
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight = 1.0
            constraint.joint_constraints.append(jc)

        goal.request.goal_constraints.append(constraint)
        return self._send_goal_blocking(goal, 'arm')

    def move_gripper(self, position: float) -> tuple[bool, str]:
        pos = np.clip(position, GRIPPER_MIN, GRIPPER_MAX)

        goal = MoveGroup.Goal()
        goal.request.group_name = 'xarm_gripper'

        constraint = Constraints()
        jc = JointConstraint()
        jc.joint_name = GRIPPER_JOINT_NAME
        jc.position = pos
        jc.tolerance_above = 0.01
        jc.tolerance_below = 0.01
        jc.weight = 1.0
        constraint.joint_constraints.append(jc)

        goal.request.goal_constraints.append(constraint)
        return self._send_goal_blocking(goal, 'gripper')

    def _send_goal_blocking(self, goal_msg, desc: str) -> tuple[bool, str]:
        self.client.wait_for_server()
        future = self.client.send_goal_async(goal_msg)
        self.executor.spin_until_future_complete(future)
        handle = future.result()

        if not handle.accepted:
            return False, f'{desc} goal rejected'

        result_future = handle.get_result_async()
        self.executor.spin_until_future_complete(result_future)
        result = result_future.result().result

        if result.error_code.val == 1:
            return True, ''
        else:
            return False, f'{desc} failed (error code={result.error_code.val})'

    def plan_and_execute(self, joint_goal: List[float], gripper_cmd: float) -> tuple[bool, str]:
        success, msg = self.move_arm(joint_goal)
        if not success:
            return False, msg

        success, msg = self.move_gripper(gripper_cmd)
        return (success, msg)

    def go_home(self) -> tuple[bool, str]:
        home_joints = [0.0] * 6
        return self.plan_and_execute(home_joints, 0.0)


if __name__ == '__main__':
    import rclpy
    from rclpy.executors import SingleThreadedExecutor

    rclpy.init()
    node = rclpy.create_node('moveit_test')
    executor = SingleThreadedExecutor()
    executor.add_node(node)

    controller = MoveItController(node, executor)
    target = [math.radians(50), math.radians(-30), math.radians(-90), 0.0, math.radians(45), 0.0]
    success, msg = controller.plan_and_execute(target, 0.85)

    log = node.get_logger()
    log.info("🎉 控制成功") if success else log.error(f"控制失敗：{msg}")

    node.destroy_node()
    rclpy.shutdown()
