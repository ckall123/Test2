import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import numpy as np

class JointController(Node):
    def __init__(self, node_name='joint_controller'):
        super().__init__(node_name)
        # Action clients
        self.arm_client = ActionClient(self, FollowJointTrajectory,
                                       '/xarm6_traj_controller/follow_joint_trajectory')
        self.gripper_client = ActionClient(self, FollowJointTrajectory,
                                           '/xarm_gripper_traj_controller/follow_joint_trajectory')
        self.arm_joint_names = ['joint1','joint2','joint3','joint4','joint5','joint6']
        self.gripper_joint_name = 'drive_joint'

    def _send_traj(self, joint_names, positions, duration=1.0, client: ActionClient=None):
        if client is None:
            raise RuntimeError('Action client is None')
        if not client.wait_for_server(timeout_sec=2.0):
            raise RuntimeError('Action server not available')

        traj = JointTrajectory()
        traj.joint_names = joint_names
        pt = JointTrajectoryPoint()
        pt.positions = [float(p) for p in positions]
        pt.time_from_start = Duration(sec=int(duration), nanosec=int((duration%1)*1e9))
        traj.points.append(pt)

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj

        send_future = client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future)
        goal_handle = send_future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('Goal rejected')
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        return True

    def send_joint_positions(self, positions, duration=1.0):
        if len(positions) != 6:
            raise ValueError('Expected 6 joint positions')
        return self._send_traj(self.arm_joint_names, positions, duration, self.arm_client)

    def control_gripper(self, open_ratio: float, duration=0.5):
        # 0.0=關、1.0=開；上下限請依實機/模擬微調
        MIN_POS, MAX_POS = 0.0, 0.8   # ← 若你的夾爪不同，調這裡
        tgt = float(np.clip(open_ratio, 0.0, 1.0)) * (MAX_POS - MIN_POS) + MIN_POS
        return self._send_traj([self.gripper_joint_name], [tgt], duration, self.gripper_client)
