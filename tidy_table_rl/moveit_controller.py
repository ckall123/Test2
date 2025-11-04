# ✅ MoveIt Controller v2 - With Exposed Limits + Fast Fallback Planning

import math
import threading
from typing import List, Dict, Tuple

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

# ✅ Joint limits as NumPy array (for fast normalization)
JOINT_LIMITS = np.array([
    (-math.radians(178), math.radians(178)),
    (-math.radians(118), math.radians(120)),
    (-math.radians(178), math.radians(11)),
    (-math.radians(178), math.radians(178)),
    (-math.radians(97),  math.radians(178)),
    (-math.radians(178), math.radians(178))
], dtype=np.float32)


class MoveItController:
    """ROS2 MoveIt controller for xArm6 + gripper."""
    def __init__(self, node: Node, executor: Executor):
        self.node = node
        self.executor = executor
        self.client = ActionClient(node, MoveGroup, '/move_action')

        self.joint_limits = JOINT_LIMITS  # ✅ exposed for normalization
        self.joint_names = ARM_JOINT_NAMES

        # --- Joint state listener ---
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
            raise RuntimeError("❌ joint_states timeout. Check if controller is publishing.")

    def get_joint_positions(self) -> np.ndarray:
        self._wait_joint_state()
        with self._joint_state_lock:
            return np.array([self._joint_state_map.get(name, 0.0) for name in self.joint_names], dtype=np.float32)

    def get_joint_velocities(self) -> np.ndarray:
        self._wait_joint_state()
        with self._joint_state_lock:
            # Use velocity if available, else return zeros
            return np.array([self._joint_state_map.get(name + '_velocity', 0.0) for name in self.joint_names], dtype=np.float32)

    def get_gripper_state(self) -> float:
        self._wait_joint_state()
        with self._joint_state_lock:
            pos = self._joint_state_map.get(GRIPPER_JOINT_NAME, GRIPPER_MIN)
            norm = (pos - GRIPPER_MIN) / (GRIPPER_MAX - GRIPPER_MIN)
            return float(np.clip(norm, 0.0, 1.0))

    def move_arm(self, joint_positions: List[float], timeout: float = 1.0) -> Tuple[bool, str]:
        if len(joint_positions) != len(self.joint_names):
            return False, "❌ Invalid joint count"

        goal = MoveGroup.Goal()
        goal.request.group_name = 'xarm6'
        goal.request.allowed_planning_time = timeout  # ✅ fast fallback

        constraint = Constraints()
        for i, (name, pos) in enumerate(zip(self.joint_names, joint_positions)):
            min_limit, max_limit = self.joint_limits[i]
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

    def move_gripper(self, position: float, timeout: float = 1.0) -> Tuple[bool, str]:
        pos = np.clip(position, GRIPPER_MIN, GRIPPER_MAX)

        goal = MoveGroup.Goal()
        goal.request.group_name = 'xarm_gripper'
        goal.request.allowed_planning_time = timeout

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

    def plan_and_execute(self, joint_goal: List[float], gripper_cmd: float, timeout: float = 1.0) -> Tuple[bool, str]:
        success, msg = self.move_arm(joint_goal, timeout)
        if not success:
            return False, msg

        return self.move_gripper(gripper_cmd, timeout)

    def go_home(self) -> Tuple[bool, str]:
        home_joints = [0.0] * 6
        return self.plan_and_execute(home_joints, 0.0)

    def _send_goal_blocking(self, goal_msg, desc: str) -> Tuple[bool, str]:
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


if __name__ == '__main__':
    rclpy.init()
    node = rclpy.create_node('moveit_test')
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)

    controller = MoveItController(node, executor)
    target = [math.radians(90), math.radians(-118), math.radians(0), 0.0, math.radians(45), 0.0]
    success, msg = controller.plan_and_execute(target, 0.85)

    log = node.get_logger()
    log.info("🎉 Success") if success else log.error(f"Fail: {msg}")

    node.destroy_node()
    rclpy.shutdown()