import rclpy
from rclpy.node import Node
from moveit_msgs.srv import GetPositionFK
from sensor_msgs.msg import JointState
import numpy as np

class FKClient(Node):
    """
    Forward Kinematics 客戶端模組：
    - 輸入 joint_state，取得末端執行器 (EE) 的 XYZ 位置
    - 預設連到 /compute_fk 服務
    """
    def __init__(self, link_name: str = "link_tcp", base_frame: str = "link_base", node_name: str = "fk_client"):
        super().__init__(node_name)
        self.link_name = link_name
        self.base_frame = base_frame

        self.cli = self.create_client(GetPositionFK, "/compute_fk")
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for FK service...")

    def compute_fk(self, joint_names: list[str], joint_positions: list[float]) -> np.ndarray:
        req = GetPositionFK.Request()
        req.header.frame_id = self.base_frame
        req.fk_link_names = [self.link_name]
        req.robot_state.joint_state.name = joint_names
        req.robot_state.joint_state.position = joint_positions

        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        if future.result() and future.result().pose_stamped:
            pos = future.result().pose_stamped[0].pose.position
            return np.array([pos.x, pos.y, pos.z], dtype=np.float32)
        else:
            self.get_logger().warn("FK call failed or returned empty result.")
            return np.zeros(3, dtype=np.float32)
