from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import time

class JointController(Node):
    """
    Joint Controller 模組：將 joint positions 發送到 trajectory topic
    預設控制 xarm6 的 joint controller
    """
    def __init__(self, topic: str = "/xarm6_traj_controller/joint_trajectory", joint_names: list[str] = None, node_name: str = "joint_controller"):
        super().__init__(node_name)
        self.publisher = self.create_publisher(JointTrajectory, topic, 10)
        self.joint_names = joint_names or [
            "joint1", "joint2", "joint3", "joint4", "joint5", "joint6"
        ]
        self.get_logger().info(f"Trajectory publisher ready on {topic}")

    def send_joint_positions(self, positions: list[float], duration: float = 1.0):
        msg = JointTrajectory()
        msg.joint_names = self.joint_names

        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start.sec = int(duration)
        point.time_from_start.nanosec = int((duration - int(duration)) * 1e9)

        msg.points.append(point)
        msg.header.stamp = self.get_clock().now().to_msg()
        self.publisher.publish(msg)

        time.sleep(duration)  # wait to finish before continuing (optional)
