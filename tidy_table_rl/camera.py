import numpy as np
from sensor_msgs.msg import Image
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from cv_bridge import CvBridge


class TopDownCamera:
    """
    接收指定 ROS2 相機主題，提供最新影像與狀態查詢。
    """
    def __init__(self, node, topic: str = '/top_camera/image_raw', resolution: tuple = (640, 480)):
        self.node = node
        self.topic = topic
        self.resolution = resolution
        self.bridge = CvBridge()
        self.latest_image: np.ndarray | None = None

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.node.create_subscription(Image, self.topic, self._image_callback, qos)
        self.node.get_logger().info(f"📷 訂閱相機主題: {self.topic}")

    def _image_callback(self, msg: Image):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        self.latest_image = img

    def get_latest_frame(self, executor, retries: int = 10) -> np.ndarray:
        """
        嘗試取得最新影像，若失敗則回傳黑圖。
        """
        for _ in range(retries):
            executor.spin_once(timeout_sec=0.1)
            if self.latest_image is not None:
                return self.latest_image

        h, w = self.resolution
        return np.zeros((h, w, 3), dtype=np.uint8)

    def is_image_ready(self) -> bool:
        """
        是否有收到影像資料。
        """
        return self.latest_image is not None

    def get_resolution(self) -> tuple[int, int]:
        """
        回傳解析度設定。
        """
        return self.resolution


# --- 測試入口 ---
if __name__ == '__main__':
    import cv2
    import rclpy
    from rclpy.executors import SingleThreadedExecutor

    rclpy.init()
    node = rclpy.create_node('top_down_camera_test')
    executor = SingleThreadedExecutor()
    executor.add_node(node)

    camera = TopDownCamera(node)
    frame = camera.get_latest_frame(executor)

    if camera.is_image_ready():
        h, w = frame.shape[:2]
        print(f"✅ 圖片擷取成功，解析度: {w}x{h}")
        cv2.imwrite("topdown_view.jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        print("已儲存圖片: topdown_view.jpg")
    else:
        print("⚠️ 未擷取到畫面，回傳為黑圖。")

    node.destroy_node()
    rclpy.shutdown()
