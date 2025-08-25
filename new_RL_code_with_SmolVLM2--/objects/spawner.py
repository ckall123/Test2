import uuid
import random
import numpy as np
import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SpawnEntity, DeleteEntity

# -------------------------------------------------
# 桌面生成範圍（與 TABLE_BOUNDS 對齊）
# -------------------------------------------------
X_RANGE = [-0.75, 0.75]
Y_RANGE = [-0.40, 0.40]
Z_HEIGHT = 1.015  # 物件 z 放在桌面高度，按需微調

# 手臂佔用區域（避免在此生成）
Arm_X_RANGE = (-0.05, 0.257)
Arm_Y_RANGE = (-0.116, 0.116)
Arm_Z_RANGE = (-0.05, 0.601)

# 可供生成的模型名稱（按需擴充）
OBJECT_NAMES = ["beer"]


class Spawner(Node):
    """Gazebo Entity Spawner（持久化使用建議）。

    - 避免把物件生成在手臂的 XY 投影盒內。
    - 兩階段 spawn：直接嘗試 → 若失敗刪同名再試一次。
    """

    def __init__(self, *, arm_bounds: dict | None = None, node_name: str | None = None):
        node_name = node_name or f"spawner_node_{uuid.uuid4().hex[:8]}"
        super().__init__(node_name)

        self.spawn_cli = self.create_client(SpawnEntity, '/spawn_entity')
        self.delete_cli = self.create_client(DeleteEntity, '/delete_entity')

        self.arm_bounds = arm_bounds or {
            'x': Arm_X_RANGE,
            'y': Arm_Y_RANGE,
            'z': Arm_Z_RANGE,
        }

        # 服務等待（短暫重試，避免卡死）
        for _ in range(5):
            if self.spawn_cli.wait_for_service(timeout_sec=1.0) and \
               self.delete_cli.wait_for_service(timeout_sec=1.0):
                break

    def spawn_object(self, name: str, position: np.ndarray, timeout: float = 10.0) -> tuple[str, bool]:
        """在指定位置 spawn 一個模型，失敗則刪同名後重試一次。
        回傳：(name, success)
        """
        if position.shape != (3,):
            position = np.asarray(position).reshape(3,)

        req = SpawnEntity.Request()
        req.name = name
        req.xml = f"""
        <sdf version='1.6'>
          <model name='{name}'>
            <include>
              <uri>model://{name}</uri>
            </include>
          </model>
        </sdf>
        """
        req.robot_namespace = name
        req.initial_pose.position.x = float(position[0])
        req.initial_pose.position.y = float(position[1])
        req.initial_pose.position.z = float(position[2])
        req.initial_pose.orientation.w = 1.0

        future = self.spawn_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        if future.done() and future.exception() is None and future.result().success:
            return name, True

        # 若失敗：刪同名後再試一次
        try:
            self.delete_object(name)
        except Exception:
            pass
        future = self.spawn_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        ok = future.done() and future.exception() is None and future.result().success
        return name, bool(ok)

    def delete_object(self, name: str, timeout: float = 8.0) -> bool:
        req = DeleteEntity.Request()
        req.name = name
        future = self.delete_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        return bool(future.done() and future.exception() is None)

    def get_safe_random_position(self, max_attempts: int = 100) -> np.ndarray:
        """回傳 X/Y 在桌面內、且不落在手臂 XY 盒內的隨機 [x,y,z]。"""
        arm = self.arm_bounds
        x0, x1 = X_RANGE
        y0, y1 = Y_RANGE
        for _ in range(max_attempts):
            x = random.uniform(x0, x1)
            y = random.uniform(y0, y1)
            if arm is None or not (arm['x'][0] <= x <= arm['x'][1] and arm['y'][0] <= y <= arm['y'][1]):
                return np.array([x, y, Z_HEIGHT], dtype=np.float32)
        # fallback：左下角
        return np.array([x0, y0, Z_HEIGHT], dtype=np.float32)