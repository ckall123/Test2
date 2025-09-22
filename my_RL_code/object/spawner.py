# object/spawner.py
import uuid
import random
import numpy as np
import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SpawnEntity, DeleteEntity
from typing import List, Tuple, Optional

# 桌面生成範圍
X_RANGE = [-1.05, 0.45]
Y_RANGE = [-1.20, -0.40]
Z_HEIGHT = 1.015

# 機械臂佔用空間，避免生成在該範圍內
ARM_X = (-0.45, -0.11)
ARM_Y = (-0.67, -0.26)


DEFAULT_MODELS = ["coke_can"]
# DEFAULT_MODELS = ["beer", "coke_can", "wood_cube_2_5cm", "wood_cube_5cm", "wood_cube_7_5cm"]
# DEFAULT_MODELS = ["beer", "bowl", "wood_cube_2_5cm", "wood_cube_5cm", "wood_cube_7_5cm", "plastic_cup","coke_can", round_tin_base, round_tin_top ]

class Spawner(Node):
    """
    隨機物件生成器，用於 Gazebo 模擬器：
    - spawn_random_objects: 隨機生成多個物件，避免重疊與手臂碰撞區。
    - delete_object / delete_all: 清除指定或所有生成的物件。
    """

    def __init__(self, object_names: Optional[List[str]] = None):
        super().__init__(f"spawner_node_{uuid.uuid4().hex[:6]}")

        self.spawn_cli = self.create_client(SpawnEntity, "/spawn_entity")
        self.delete_cli = self.create_client(DeleteEntity, "/delete_entity")
        self.object_names = object_names or DEFAULT_MODELS
        self.spawned_names: List[str] = []

        for _ in range(5):
            if self.spawn_cli.wait_for_service(timeout_sec=1.0) and self.delete_cli.wait_for_service(timeout_sec=1.0):
                break

    def delete_object(self, name: str) -> bool:
        req = DeleteEntity.Request(name=name)
        future = self.delete_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)
        if name in self.spawned_names:
            self.spawned_names.remove(name)
        return future.done() and not future.exception()

    def delete_all(self):
        for name in self.spawned_names[:]:
            self.delete_object(name)
        self.spawned_names.clear()

    def _random_position(self, placed: List[np.ndarray], min_dist: float = 0.08) -> np.ndarray:
        for _ in range(100):
            x = random.uniform(*X_RANGE)
            y = random.uniform(*Y_RANGE)
            if ARM_X[0] <= x <= ARM_X[1] and ARM_Y[0] <= y <= ARM_Y[1]:
                continue
            pos = np.array([x, y, Z_HEIGHT], dtype=np.float32)
            if all(np.linalg.norm(pos[:2] - p[:2]) >= min_dist for p in placed):
                return pos
        return np.array([X_RANGE[0], Y_RANGE[0], Z_HEIGHT], dtype=np.float32)

    def _spawn(self, model: str, position: np.ndarray, name: Optional[str] = None) -> Optional[str]:
        instance = name or f"{model}_{uuid.uuid4().hex[:6]}"
        req = SpawnEntity.Request()
        req.name = instance
        req.xml = f"""
        <sdf version='1.6'>
          <model name='{instance}'>
            <include>
              <uri>model://{model}</uri>
            </include>
          </model>
        </sdf>"""
        req.robot_namespace = instance
        req.initial_pose.position.x = float(position[0])
        req.initial_pose.position.y = float(position[1])
        req.initial_pose.position.z = float(position[2])
        req.initial_pose.orientation.w = 1.0

        future = self.spawn_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        success = future.done() and not future.exception() and future.result().success
        if success:
            self.spawned_names.append(instance)
            return instance
        return None

    def spawn_random_objects(self, count: int, avoid_dist: float = 0.08, unique: bool = True) -> List[Tuple[str, np.ndarray]]:
        models = self.object_names.copy()
        if unique and count > len(models):
            self.get_logger().warn("要求數量超過可用模型數，將進行裁剪。")
            count = len(models)

        bases = random.sample(models, count) if unique else [random.choice(models) for _ in range(count)]
        results = []
        placed = []

        for model in bases:
            pos = self._random_position(placed, min_dist=avoid_dist)
            name = self._spawn(model, pos)
            if name:
                results.append((name, pos))
                placed.append(pos)

        return results


# 測試：生成 3 個隨機物件並在 10 秒後刪除
if __name__ == '__main__':
    import time

    rclpy.init()
    spawner = Spawner()

    try:
        print("🎯 嘗試生成 3 個物件...")
        objects = spawner.spawn_random_objects(count=1)
        for name, pos in objects:
            print(f"✅ 已生成: {name} at {pos.round(3).tolist()}")

        # time.sleep(10)  # 等一下，讓你看看物件
        # print("🧹 開始刪除所有生成的物件...")
        # spawner.delete_all()

    finally:
        spawner.destroy_node()
        rclpy.shutdown()
        print("🛑 測試結束")
