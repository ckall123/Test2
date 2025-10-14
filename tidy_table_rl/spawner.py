import json
import random
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
from geometry_msgs.msg import Pose
from gazebo_msgs.srv import SpawnEntity, DeleteEntity

CONFIG_PATH = Path("config.json")

# 桌面範圍
X_RANGE = [-1.05, 0.45]
Y_RANGE = [-1.20, -0.40]
Z_HEIGHT = 1.015

# 機械臂佔用區域
ARM_X = (-0.45, -0.11)
ARM_Y = (-0.67, -0.26)

DEFAULT_MODELS = ["coke_can", "wood_cube_5cm"]


class Spawner:
    def __init__(self, node, executor, models: Optional[List[str]] = None):
        self.node = node
        self.executor = executor
        self.cli_spawn = self.node.create_client(SpawnEntity, '/spawn_entity')
        self.cli_delete = self.node.create_client(DeleteEntity, '/delete_entity')
        self.object_names = models or DEFAULT_MODELS
        self.spawned_names: List[str] = []

        for _ in range(5):
            if self.cli_spawn.wait_for_service(timeout_sec=1.0) and self.cli_delete.wait_for_service(timeout_sec=1.0):
                break

    def _load_config(self):
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH, 'r') as f:
                return json.load(f)
        return {}

    def _save_config(self, data):
        with open(CONFIG_PATH, 'w') as f:
            json.dump(data, f, indent=2)

    def _update_config_add(self, name):
        cfg = self._load_config()
        cfg.setdefault("target_objects", []).append(name)
        cfg["target_objects"] = list(set(cfg["target_objects"]))  # 去重
        self._save_config(cfg)

    def _update_config_remove(self, name):
        cfg = self._load_config()
        if "target_objects" in cfg and name in cfg["target_objects"]:
            cfg["target_objects"].remove(name)
            self._save_config(cfg)

    def delete_all(self):
        for name in self.spawned_names[:]:
            self.delete(name)
        self.spawned_names.clear()
        for _ in range(5):
            self.executor.spin_once(timeout_sec=0.1)

    def delete(self, name: str):
        req = DeleteEntity.Request()
        req.name = name
        self.cli_delete.call_async(req)
        self._update_config_remove(name)

    def _is_valid(self, x: float, y: float) -> bool:
        return not (ARM_X[0] <= x <= ARM_X[1] and ARM_Y[0] <= y <= ARM_Y[1])

    def spawn_model(self, unique_name: str, model_name: str, pose: Pose):
        model_path = Path.home() / '.gazebo' / 'models' / model_name / 'model.sdf'
        if not model_path.exists():
            self.node.get_logger().error(f'Model file not found: {model_path}')
            return False

        xml_string = model_path.read_text()

        req = SpawnEntity.Request()
        req.name = unique_name
        req.xml = xml_string
        req.initial_pose = pose

        future = self.cli_spawn.call_async(req)
        self.executor.spin_until_future_complete(future)

        success = future.done() and not future.exception() and future.result().success
        if success:
            self.spawned_names.append(unique_name)
            self._update_config_add(unique_name)
            self.node.get_logger().info(f"✅ Spawned {unique_name} at ({pose.position.x:.2f}, {pose.position.y:.2f})")
            return True
        else:
            self.node.get_logger().warn(f"⚠️ Failed to spawn {unique_name}")
            return False

    def spawn_random_objects(self, count: int = 1) -> List[Dict]:
        name_counts = {}
        spawned_info = []

        for _ in range(count):
            model = random.choice(self.object_names)
            name_counts[model] = name_counts.get(model, 0) + 1
            suffix = f"_{name_counts[model]}"
            unique_name = f"{model}{suffix}"

            for _ in range(100):
                x = random.uniform(*X_RANGE)
                y = random.uniform(*Y_RANGE)
                if self._is_valid(x, y):
                    break
            else:
                self.node.get_logger().warn(f"找不到有效位置給 {unique_name}，跳過。")
                continue

            pose = Pose()
            pose.position.x = x
            pose.position.y = y
            pose.position.z = Z_HEIGHT

            if self.spawn_model(unique_name, model, pose):
                spawned_info.append({
                    "name": unique_name,
                    "model": model,
                    "pose": pose
                })

        for _ in range(5):
            self.executor.spin_once(timeout_sec=0.1)

        return spawned_info


if __name__ == '__main__':
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node

    rclpy.init()
    executor = SingleThreadedExecutor()
    node = Node('spawner_test_node')
    executor.add_node(node)

    spawner = Spawner(node=node, executor=executor)

    try:
        node.get_logger().info("🎯 嘗試生成 3 個物件...")
        spawner.spawn_random_objects(count=3)
        input("🔍 按下 Enter 鍵測試刪除這些物件...\n")
        spawner.delete_all()
    finally:
        node.destroy_node()
        rclpy.shutdown()
        print("🛑 測試結束")
