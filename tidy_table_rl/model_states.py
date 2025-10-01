# env/model_states.py
import rclpy
from rclpy.node import Node
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Pose
from typing import Dict, Optional, Tuple

class ModelStateTracker(Node):
    def __init__(self, topic: str = "/gazebo/model_states"):
        super().__init__("model_state_tracker")
        self._poses: Dict[str, Pose] = {}
        self._sub = self.create_subscription(ModelStates, topic, self._cb, 10)
        self._got_first = False

    def _cb(self, msg: ModelStates):
        for name, pose in zip(msg.name, msg.pose):
            self._poses[name] = pose
        self._got_first = True

    def wait_for_first(self, timeout: float = 5.0) -> bool:
        """阻塞直到收到第一筆 model states（最多 timeout 秒）"""
        import time
        t0 = time.time()
        while not self._got_first and (time.time() - t0) < timeout:
            rclpy.spin_once(self, timeout_sec=0.05)
        return self._got_first

    def get_xy(self, model_name: str) -> Optional[Tuple[float, float]]:
        p = self._poses.get(model_name)
        if p is None:
            return None
        return float(p.position.x), float(p.position.y)

    def get_pose(self, model_name: str) -> Optional[Pose]:
        return self._poses.get(model_name)
