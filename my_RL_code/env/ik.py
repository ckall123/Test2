import rclpy
from rclpy.node import Node
from moveit_msgs.srv import GetPositionIK
from geometry_msgs.msg import PoseStamped
import numpy as np

class IKClient(Node):
    """
    Inverse Kinematics 客戶端模組：
    - 輸入末端位置 XYZ，回傳 joint_angles（若可解）
    - 使用 MoveIt 的 /compute_ik 服務
    """
    def __init__(self, group_name: str = "xarm6", link_name: str = "link_tcp", node_name: str = "ik_client"):
        super().__init__(node_name)
        self.group_name = group_name
        self.link_name = link_name

        self.cli = self.create_client(GetPositionIK, "/compute_ik")
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for IK service...")

    def compute_ik(self, xyz: np.ndarray) -> list[float] | None:
        req = GetPositionIK.Request()
        req.ik_request.group_name = self.group_name
        req.ik_request.ik_link_name = self.link_name
        req.ik_request.pose_stamped = PoseStamped()
        req.ik_request.pose_stamped.header.frame_id = "link_base"
        req.ik_request.pose_stamped.pose.position.x = float(xyz[0])
        req.ik_request.pose_stamped.pose.position.y = float(xyz[1])
        req.ik_request.pose_stamped.pose.position.z = float(xyz[2])
        req.ik_request.pose_stamped.pose.orientation.w = 1.0  # 無轉動

        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        if future.result() and future.result().solution.joint_state.name:
            return list(future.result().solution.joint_state.position)
        else:
            self.get_logger().warn("IK failed or returned empty solution.")
            return None
