from sensor_msgs.msg import JointState
import rclpy
from rclpy.node import Node

class GripperStateTracker(Node):
    def __init__(self, joint_name: str = 'drive_joint', node_name='gripper_tracker'):
        super().__init__(node_name)
        self.joint_name = joint_name
        self.current_state = 0.0
        self.last_msg = None
        self.create_subscription(JointState, '/joint_states', self._cb, 10)

    def _cb(self, msg: JointState):
        self.last_msg = msg
        try:
            idx = msg.name.index(self.joint_name)
            self.current_state = msg.position[idx]
        except ValueError:
            self.get_logger().warn(f"'{self.joint_name}' not in joint_states")

    def get_state(self) -> float:
        return float(self.current_state)
