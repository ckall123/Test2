# ik_client.py
#!/usr/bin/env python3
from __future__ import annotations
import threading
from typing import Dict, Iterable, Optional, Sequence, Tuple

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from builtin_interfaces.msg import Duration
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import RobotState


class IKSolverClient:
    """
    可 import 的 MoveIt IK 客戶端：
    - 背景 executor 自動處理 /joint_states、service 呼叫
    - 提供 solve_pose() 取 IK；回傳 (error_code, {joint: position} 只含你指定的六軸)
    - with 介面自動 start()/stop()，也可手動呼叫
    """

    def __init__(
        self,
        group_name: str = "xarm6",
        base_frame: str = "link_base",
        tip_link: Optional[str] = "link6",  # 若你之後加了 link_tcp 可改成 "link_tcp"
        joint_names: Sequence[str] = ("joint1","joint2","joint3","joint4","joint5","joint6"),
        joint_states_topic: str = "/joint_states",
        compute_ik_service: str = "/compute_ik",
        seed_timeout_sec: float = 5.0,
        service_timeout_sec: float = 2.0,
    ) -> None:
        self.group_name = group_name
        self.base_frame = base_frame
        self.tip_link = tip_link
        self.arm_order = list(joint_names)
        self.joint_states_topic = joint_states_topic
        self.compute_ik_service = compute_ik_service
        self.seed_timeout_sec = seed_timeout_sec
        self.service_timeout_sec = float(service_timeout_sec)

        self._node: Optional[Node] = None
        self._exec: Optional[SingleThreadedExecutor] = None
        self._thread: Optional[threading.Thread] = None
        self._js_evt = threading.Event()
        self._seed_lock = threading.Lock()
        self._seed_msg: Optional[JointState] = None
        self._own_rclpy_context = False  # 若本模組負責 init，就負責 shutdown；否則不動

    # --- lifecycle ---------------------------------------------------------

    def start(self) -> "IKSolverClient":
        # rclpy.init 可能已由外部呼叫；這裡容錯處理
        try:
            if not rclpy.ok():
                rclpy.init()
                self._own_rclpy_context = True
        except Exception:
            # 已初始化就略過
            pass

        self._node = Node("ik_solver_client")
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,  # /joint_states 常見 QoS
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self._node.create_subscription(JointState, self.joint_states_topic, self._on_joint_states, qos)

        self._cli = self._node.create_client(GetPositionIK, self.compute_ik_service)
        self._node.get_logger().info(f"Waiting for {self.compute_ik_service} …")
        self._cli.wait_for_service()

        # 背景 executor
        self._exec = SingleThreadedExecutor()
        self._exec.add_node(self._node)
        self._thread = threading.Thread(target=self._spin, name="ik_client_spin", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._exec:
            self._exec.shutdown()
        if self._node:
            self._node.destroy_node()
        if self._own_rclpy_context and rclpy.ok():
            rclpy.shutdown()
        self._exec = None
        self._node = None
        self._thread = None

    def __enter__(self) -> "IKSolverClient":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def _spin(self) -> None:
        assert self._exec is not None
        try:
            self._exec.spin()
        except Exception:
            pass

    # --- subscriptions -----------------------------------------------------

    def _on_joint_states(self, msg: JointState):
        if not msg.name:
            return
        name2pos = {n: p for n, p in zip(msg.name, msg.position)}
        if any(n not in name2pos for n in self.arm_order):
            return
        js = JointState()
        js.name = self.arm_order.copy()
        js.position = [name2pos[n] for n in self.arm_order]
        with self._seed_lock:
            self._seed_msg = js
        self._js_evt.set()

    # --- public API --------------------------------------------------------

    def wait_for_seed(self, timeout_sec: Optional[float] = None) -> bool:
        if timeout_sec is None:
            timeout_sec = self.seed_timeout_sec
        if self._seed_msg is not None:
            return True
        return self._js_evt.wait(timeout=timeout_sec)

    def latest_seed(self) -> Optional[JointState]:
        with self._seed_lock:
            return self._seed_msg

    def solve_pose(
        self,
        pos_xyz: Tuple[float, float, float],
        quat_xyzw: Tuple[float, float, float, float],
        *,
        frame_id: Optional[str] = None,
        tip_link: Optional[str] = None,
        avoid_collisions: bool = False,
        timeout_sec: Optional[float] = None,
    ) -> Tuple[int, Dict[str, float]]:
        """
        呼叫 /compute_ik。回傳 (error_code, 解出的 {joint: position} 只含 arm_order)。
        error_code == 1 表成功；其他值如 -31 表無解。
        """
        if self._node is None:
            raise RuntimeError("IKSolverClient not started. Call start() or use context manager.")
        if not self.wait_for_seed():
            raise TimeoutError(f"No /joint_states within {self.seed_timeout_sec}s.")

        seed = self.latest_seed()
        assert seed is not None

        ps = PoseStamped()
        ps.header.frame_id = frame_id or self.base_frame
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = pos_xyz
        ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z, ps.pose.orientation.w = quat_xyzw

        req = GetPositionIK.Request()
        req.ik_request.group_name = self.group_name
        req.ik_request.pose_stamped = ps
        req.ik_request.avoid_collisions = bool(avoid_collisions)
        secs = float(timeout_sec if timeout_sec is not None else self.service_timeout_sec)
        req.ik_request.timeout = Duration(sec=int(secs), nanosec=int((secs % 1.0) * 1e9))
        use_tip = tip_link if tip_link is not None else self.tip_link
        if use_tip:
            req.ik_request.ik_link_name = use_tip

        rs = RobotState()
        rs.joint_state = seed
        req.ik_request.robot_state = rs

        fut = self._cli.call_async(req)
        # 用 node 的 executor 會處理 future，不需另外 spin
        fut.result()  # 讓 KeyboardInterrupt 可打斷
        res = fut.result()
        if res is None:
            raise RuntimeError("IK service failed")

        names = list(res.solution.joint_state.name)
        positions = list(res.solution.joint_state.position)
        n2p = {n: p for n, p in zip(names, positions)}
        filtered = {n: n2p[n] for n in self.arm_order if n in n2p}
        return res.error_code.val, filtered
