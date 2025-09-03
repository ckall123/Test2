import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2


class TopDownCamera(Node):
    """
    簡易 ROS2 相機訂閱模組：
    - 使用 cv_bridge 將 /image_raw 轉換為 numpy 格式 (HxWx3)
    - 自動 resize 為指定解析度
    - 提供 get_latest_frame() 取得最新畫面
    """

    def __init__(self, topic: str = "/top_camera/image_raw", resolution=(64, 64), node_name: str = "top_down_camera"):
        super().__init__(node_name)
        self.bridge = CvBridge()
        self.latest_frame = None
        self.resolution = resolution

        self.subscription = self.create_subscription(Image, topic, self._image_callback, 10)
        self.get_logger().info(f"Subscribed to camera topic: {topic}")

    def _image_callback(self, msg: Image):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            resized = cv2.resize(img, self.resolution, interpolation=cv2.INTER_AREA)
            self.latest_frame = resized.astype(np.uint8)
        except Exception as e:
            self.get_logger().error(f"Camera image conversion failed: {e}")

    def get_latest_frame(self) -> np.ndarray:
        return self.latest_frame.copy() if self.latest_frame is not None else None
