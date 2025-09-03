from rclpy.node import Node
from sensor_msgs.msg import JointState

class GripperStateTracker(Node):
    """
    夾爪狀態模組：訂閱 joint_states，追蹤 gripper (drive_joint) 的位置
    可用於 RL 的觀察維度中
    """
    def __init__(self, joint_name: str = "drive_joint", node_name: str = "gripper_tracker"):
        super().__init__(node_name)
        self.joint_name = joint_name
        self.current_state = 0.0

        self.subscription = self.create_subscription(JointState, "/joint_states", self._callback, 10)
        self.get_logger().info("Gripper state tracker initialized.")

    def _callback(self, msg: JointState):
        try:
            idx = msg.name.index(self.joint_name)
            self.current_state = msg.position[idx]
        except ValueError:
            self.get_logger().warn(f"Joint '{self.joint_name}' not found in joint_states.")

    def get_state(self) -> float:
        return float(self.current_state)
