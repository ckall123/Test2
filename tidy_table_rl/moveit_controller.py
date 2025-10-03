# tidy_table_rl/moveit_controller.py
# 給 XArm6 + MoveIt2 的同步控制器（手臂 + 夾爪 + 基於接觸的自動附著/釋放）

from __future__ import annotations
import math
import time
import threading

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint

# 專案現有模組
from attach_detach import attach_object, detach_object
from gripper_contact import in_contact
from ik import solve_ik, get_current_pose

JOINT_NAMES = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
JOINT_LIMITS = [
    (-math.radians(178), math.radians(178)),
    (-math.radians(118), math.radians(120)),
    (-math.radians(178), math.radians(11)),
    (-math.radians(178), math.radians(178)),
    (-math.radians(97),  math.radians(178)),
    (-math.radians(178), math.radians(178)),
]

GRIPPER_GROUP = 'xarm_gripper'
GRIPPER_JOINT = 'drive_joint'
GRIPPER_MAX = 0.8552  # radians
GRIPPER_ATTACH_THRESHOLD = 0.5  # open度 < 0.5 視為夾緊

HOME_JOINTS = [  # 依你測過的姿態當 home
    math.radians(-176),
    math.radians(-21),
    math.radians(-4),
    math.radians(0),
    math.radians(25),
    math.radians(-176),
]


class MoveIt2Controller:
    """提供 MoveIt2 控制 arm+gripper 的同步 API，並在合適時機自動附著/釋放。"""

    def __init__(self,
                 arm_group: str = 'xarm6',
                 eef_link: str = 'link6',
                 wait_timeout: float = 3.0):
        if not rclpy.ok():
            rclpy.init(args=None)
        self.node = Node('moveit2_controller')
        self.arm_group = arm_group
        self.eef_link = eef_link
        self.wait_timeout = wait_timeout

        # 行動客戶端
        self.arm_client = ActionClient(self.node, MoveGroup, '/move_action')
        self.gripper_client = ActionClient(self.node, MoveGroup, '/move_action')

        # 同步事件
        self._arm_done_evt = threading.Event()
        self._grip_done_evt = threading.Event()

        # 狀態
        self.held_object: str | None = None
        self._last_grip_pos = 0.0

    # -------------------- 公用：收尾 --------------------

    def destroy_node(self):
        try:
            self.node.destroy_node()
        finally:
            if rclpy.ok():
                rclpy.shutdown()

    # -------------------- Arm：關節與姿態 --------------------

    def move_arm(self, joint_positions: list[float], tol: float = 0.01) -> bool:
        """以 JointConstraint 規劃到位，阻塞等待結果。"""
        if len(joint_positions) != 6:
            self.node.get_logger().error("需要 6 維關節角度")
            return False

        # 上下限檢查
        for i, ang in enumerate(joint_positions):
            lo, hi = JOINT_LIMITS[i]
            if not (lo <= ang <= hi):
                self.node.get_logger().error(f'❌ {JOINT_NAMES[i]} 超出範圍: {ang:.3f} rad')
                return False

        if not self.arm_client.wait_for_server(timeout_sec=self.wait_timeout):
            self.node.get_logger().error("MoveIt Arm server 不可用")
            return False

        goal = MoveGroup.Goal()
        goal.request.group_name = self.arm_group
        # （可按需設定速度/加速度比例等參數）
        # goal.request.max_velocity_scaling_factor = 0.2
        # goal.request.max_acceleration_scaling_factor = 0.2

        cons = Constraints()
        for name, ang in zip(JOINT_NAMES, joint_positions):
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = float(ang)
            jc.tolerance_above = tol
            jc.tolerance_below = tol
            jc.weight = 1.0
            cons.joint_constraints.append(jc)
        goal.request.goal_constraints.append(cons)

        # 送出並等待
        self._arm_done_evt.clear()
        self.arm_client.send_goal_async(goal).add_done_callback(self._arm_goal_cb)
        while not self._arm_done_evt.wait(timeout=0.1):
            rclpy.spin_once(self.node, timeout_sec=0.05)
        return self._arm_success

    def _arm_goal_cb(self, fut):
        handle = fut.result()
        if not handle or not handle.accepted:
            self._arm_success = False
            self.node.get_logger().error("❌ 手臂目標被拒絕")
            self._arm_done_evt.set()
            return
        handle.get_result_async().add_done_callback(self._arm_result_cb)

    def _arm_result_cb(self, fut):
        res = fut.result().result
        ok = bool(res and res.error_code.val == 1)
        self._arm_success = ok
        self.node.get_logger().info('🎉 手臂移動完成' if ok else f'⚠️ 手臂移動失敗：{getattr(res.error_code, "val", -1)}')
        self._arm_done_evt.set()

    def move_to_pose(self, pos_xyz: list[float], quat_xyzw: list[float], duration: float | None = None) -> bool:
        """利用 IK 把姿態轉 6R 關節後下達。"""
        code, sol6, _seed = solve_ik(pos_xyz, quat_xyzw, group=self.arm_group, tip=self.eef_link)
        if code != 1:
            self.node.get_logger().error(f"❌ IK 失敗，error_code={code}")
            return False
        return self.move_arm(sol6)

    def move_arm_home(self) -> bool:
        return self.move_arm(HOME_JOINTS)

    def get_current_pose(self) -> tuple[list[float] | None, list[float] | None]:
        """回傳 (pos_xyz, quat_xyzw)；失敗回 (None, None)。"""
        return get_current_pose(base="link_base", tip=self.eef_link)

    # -------------------- Gripper：位置控制 + 自動附著 --------------------

    def control_gripper(self, g_open: float, duration: float | None = None) -> bool:
        """
        控制夾爪開合（0=關緊，1=全開）。自動判斷是否需要 attach/detach。
        - 當 g_open < 0.5 且雙指同時接觸到同一物時 → attach
        - 當 g_open >= 0.5 且目前有附著 → detach
        """
        target = max(0.0, min(1.0, float(g_open))) * GRIPPER_MAX
        ok = self._set_gripper_joint(target)

        # 自動附著/釋放（給模擬用；實機請小心）
        try:
            if target < GRIPPER_ATTACH_THRESHOLD * GRIPPER_MAX:
                obj = in_contact()  # 若兩側同時接觸到同一物件，回傳名字
                if obj and not self.held_object:
                    attach_object(obj)
                    self.held_object = obj
                    self.node.get_logger().info(f"🔗 attach: {obj}")
            else:
                if self.held_object:
                    detach_object(self.held_object)
                    self.node.get_logger().info(f"🔓 detach: {self.held_object}")
                    self.held_object = None
        except Exception as e:
            self.node.get_logger().warn(f"attach/detach 例外：{e}")

        self._last_grip_pos = target
        return ok

    def open_gripper(self) -> bool:
        return self.control_gripper(1.0)

    def _set_gripper_joint(self, position_rad: float, tol: float = 0.01) -> bool:
        """以 JointConstraint 控制 gripper 的 drive_joint。"""
        if not self.gripper_client.wait_for_server(timeout_sec=self.wait_timeout):
            self.node.get_logger().error("MoveIt Gripper server 不可用")
            return False

        goal = MoveGroup.Goal()
        goal.request.group_name = GRIPPER_GROUP

        cons = Constraints()
        jc = JointConstraint()
        jc.joint_name = GRIPPER_JOINT
        jc.position = float(position_rad)
        jc.tolerance_above = tol
        jc.tolerance_below = tol
        jc.weight = 1.0
        cons.joint_constraints.append(jc)
        goal.request.goal_constraints.append(cons)

        self._grip_done_evt.clear()
        self.gripper_client.send_goal_async(goal).add_done_callback(self._grip_goal_cb)
        while not self._grip_done_evt.wait(timeout=0.1):
            rclpy.spin_once(self.node, timeout_sec=0.05)
        return self._grip_success

    def _grip_goal_cb(self, fut):
        handle = fut.result()
        if not handle or not handle.accepted:
            self._grip_success = False
            self.node.get_logger().error("❌ 夾爪動作被拒絕")
            self._grip_done_evt.set()
            return
        handle.get_result_async().add_done_callback(self._grip_result_cb)

    def _grip_result_cb(self, fut):
        res = fut.result().result
        ok = bool(res and res.error_code.val == 1)
        self._grip_success = ok
        self.node.get_logger().info('🎉 夾爪動作完成' if ok else f'⚠️ 夾爪動作失敗：{getattr(res.error_code, "val", -1)}')
        self._grip_done_evt.set()
