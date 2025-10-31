import json
import random
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
from geometry_msgs.msg import Pose
from gazebo_msgs.srv import SpawnEntity, DeleteEntity

CONFIG_PATH = Path("config.json")
X_RANGE = [-1.05, 0.45]
Y_RANGE = [-1.20, -0.40]
Z_HEIGHT = 1.015
ARM_X = (-0.45, -0.11)
ARM_Y = (-0.67, -0.26)
DEFAULT_MODELS = ["coke_can", "wood_cube_5cm"]


class Spawner:
    def __init__(self, node, executor, models: Optional[List[str]] = None):
        self.node = node
        self.executor = executor
        self.cli_spawn = node.create_client(SpawnEntity, '/spawn_entity')
        self.cli_delete = node.create_client(DeleteEntity, '/delete_entity')
        self.object_models = models or DEFAULT_MODELS
        self.spawned: List[str] = []

        for _ in range(5):
            if self.cli_spawn.wait_for_service(timeout_sec=1.0) and self.cli_delete.wait_for_service(timeout_sec=1.0):
                break

    def _load_config(self) -> dict:
        return json.loads(CONFIG_PATH.read_text()) if CONFIG_PATH.exists() else {}

    def _save_config(self, data: dict):
        CONFIG_PATH.write_text(json.dumps(data, indent=2))

    def _update_config_add(self, name: str):
        cfg = self._load_config()
        targets = cfg.setdefault("target_objects", [])
        if name not in targets:
            targets.append(name)
            self._save_config(cfg)

    def _update_config_remove(self, name: str):
        cfg = self._load_config()
        if "target_objects" in cfg and name in cfg["target_objects"]:
            cfg["target_objects"].remove(name)
            self._save_config(cfg)

    def _is_valid(self, x: float, y: float) -> bool:
        return not (ARM_X[0] <= x <= ARM_X[1] and ARM_Y[0] <= y <= ARM_Y[1])

    def spawn_model(self, unique_name: str, model_name: str, pose: Pose) -> bool:
        model_path = Path.home() / '.gazebo' / 'models' / model_name / 'model.sdf'
        if not model_path.exists():
            self.node.get_logger().error(f"找不到模型檔案: {model_path}")
            return False

        req = SpawnEntity.Request()
        req.name = unique_name
        req.xml = model_path.read_text()
        req.initial_pose = pose

        future = self.cli_spawn.call_async(req)
        self.executor.spin_until_future_complete(future)

        success = future.done() and not future.exception() and future.result().success
        if success:
            self.spawned.append(unique_name)
            self._update_config_add(unique_name)
            self.node.get_logger().info(f"✅ 已生成 {unique_name} at ({pose.position.x:.2f}, {pose.position.y:.2f})")
        else:
            self.node.get_logger().warn(f"⚠️ 生成失敗：{unique_name}")
        return success

    def delete(self, name: str):
        req = DeleteEntity.Request()
        req.name = name
        self.cli_delete.call_async(req)
        self._update_config_remove(name)
        if name in self.spawned:
            self.spawned.remove(name)

    def delete_all(self):
        cfg = self._load_config()
        targets = cfg.get("target_objects", [])

        for name in targets:
            self.delete(name)

        self.spawned.clear()
        cfg["target_objects"] = []
        self._save_config(cfg)

        for _ in range(10):
            self.executor.spin_once(timeout_sec=0.1)


    def spawn_random_objects(self, count: int = 1) -> List[Dict]:
        spawned_info = []
        name_counter = {}

        for _ in range(count):
            model = random.choice(self.object_models)
            name_counter[model] = name_counter.get(model, 0) + 1
            unique = f"{model}_{name_counter[model]}"

            for _ in range(100):
                x = random.uniform(*X_RANGE)
                y = random.uniform(*Y_RANGE)
                if self._is_valid(x, y):
                    break
            else:
                self.node.get_logger().warn(f"⚠️ 找不到合適位置，跳過 {unique}")
                continue

            pose = Pose()
            pose.position.x = x
            pose.position.y = y
            pose.position.z = Z_HEIGHT

            if self.spawn_model(unique, model, pose):
                spawned_info.append({"name": unique, "model": model, "pose": pose})

        for _ in range(5):
            self.executor.spin_once(timeout_sec=0.1)

        return spawned_info


if __name__ == '__main__':
    import rclpy
    from rclpy.executors import SingleThreadedExecutor

    rclpy.init()
    node = rclpy.create_node('spawner_test')
    executor = SingleThreadedExecutor()
    executor.add_node(node)

    spawner = Spawner(node, executor)
    spawner.delete_all()

    try:
        node.get_logger().info("🎯 測試生成 3 個物件中...")
        spawner.spawn_random_objects(count=3)
        input("🧽 按下 Enter 清除這些物件...")
        spawner.delete_all()
    finally:
        node.destroy_node()
        rclpy.shutdown()
