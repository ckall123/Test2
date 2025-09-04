import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from cv_bridge import CvBridge
import numpy as np

class TopDownCamera(Node):
    def __init__(self, resolution=(640, 480), topic='/top_camera/image_raw', node_name='top_down_camera'):
        super().__init__(node_name)
        self.resolution = resolution
        self.bridge = CvBridge()
        self._latest = None
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST, depth=1
        )
        self.sub = self.create_subscription(Image, topic, self._cb, qos)

    def _cb(self, msg: Image):
        # 你當前來源是 rgb8 → 直接轉成 RGB ndarray
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            self._latest = img
        except Exception as e:
            self.get_logger().warn(f'cv_bridge error: {e}')

    def get_latest_frame(self):
        if self._latest is None:
            rclpy.spin_once(self, timeout_sec=0.05)
        if self._latest is None:
            # 安全 fallback：黑畫面
            w, h = self.resolution
            return np.zeros((h, w, 3), dtype=np.uint8)
        return self._latest
