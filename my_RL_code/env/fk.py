#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from moveit_msgs.srv import GetPositionFK
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose
from typing import List, Tuple, Optional
import numpy as np

class FKClient(Node):
    """
    Forward Kinematics 客戶端
    - 輸入關節名稱與角度，回傳末端執行器的位置與方向（XYZ + 四元數）
    - 預設使用 MoveIt 的 /compute_fk 服務
    """
    def __init__(self, 
                 link_name: str = "link6", 
                 base_frame: str = "link_base",
                 node_name: str = "fk_client"):
        super().__init__(node_name)
        self.link_name = link_name
        self.base_frame = base_frame
        self.cli = self.create_client(GetPositionFK, "/compute_fk")

        self.get_logger().info("等待 FK 服務啟動中...")
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("仍在等待 FK 服務...")
        self.get_logger().info("已連接到 FK 服務！")

    def compute_fk(self, 
                   joint_names: List[str], 
                   joint_positions: List[float]
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        輸入關節資訊，回傳末端位置與方向 (XYZ, 四元數)
        """
        req = GetPositionFK.Request()
        req.header.frame_id = self.base_frame
        req.fk_link_names = [self.link_name]
        req.robot_state.joint_state = JointState(name=joint_names, position=joint_positions)

        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        if not future.result() or not future.result().pose_stamped:
            self.get_logger().error("FK 查詢失敗或無結果。")
            return None

        pose: Pose = future.result().pose_stamped[0].pose
        position = np.array([pose.position.x, pose.position.y, pose.position.z], dtype=np.float32)
        orientation = np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w], dtype=np.float32)
        return position, orientation


def main():
    rclpy.init()
    fk_client = FKClient()

    # 範例：給定一組關節角度
    joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
    joint_positions = [0.1, -1.2, 0.3, 0.0, 1.0, -0.5]  # 單位: 弧度

    result = fk_client.compute_fk(joint_names, joint_positions)
    if result:
        pos, quat = result
        print("末端位置 (xyz):", pos)
        print("末端方向 (四元數):", quat)

    fk_client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
