import random
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
from geometry_msgs.msg import Pose
from gazebo_msgs.srv import SpawnEntity, DeleteEntity

# 桌面範圍
X_RANGE = [-1.05, 0.45]
Y_RANGE = [-1.20, -0.40]
Z_HEIGHT = 1.015

# 機械臂佔用區域
ARM_X = (-0.45, -0.11)
ARM_Y = (-0.67, -0.26)

DEFAULT_MODELS = ["coke_can", "wood_cube_5cm"]

# base_link 在 world 座標下的位置與旋轉（手動量測）
BASE_X = -0.200000
BASE_Y = -0.500001
BASE_Z = 1.020995
BASE_ROLL = -0.000003
BASE_PITCH = 0.0
BASE_YAW = 1.571000

USE_TF = False


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

    def delete_all(self):
        for name in self.spawned_names[:]:
            self.delete(name)
        self.spawned_names.clear()

    def delete(self, name: str):
        req = DeleteEntity.Request()
        req.name = name
        self.cli_delete.call_async(req)

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
            self.register_in_moveit(unique_name, pose.position.x, pose.position.y, pose.position.z)
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
            suffix = f"_{name_counts[model]}" if name_counts[model] > 1 else "_1"
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

        return spawned_info

    def register_in_moveit(self, name: str, x: float, y: float, z: float):
        from moveit_msgs.msg import CollisionObject, PlanningScene
        from moveit_msgs.srv import ApplyPlanningScene
        from shape_msgs.msg import SolidPrimitive
        from geometry_msgs.msg import PoseStamped
        from std_msgs.msg import Header
        import math

        client = self.node.create_client(ApplyPlanningScene, '/apply_planning_scene')
        while not client.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info('等待 /apply_planning_scene 服務中...')

        obj = CollisionObject()
        obj.id = name
        obj.header = Header(frame_id='link_base')

        if USE_TF:
            pass
        else:
            t = np.array([BASE_X, BASE_Y, BASE_Z])
            cy = math.cos(BASE_YAW)
            sy = math.sin(BASE_YAW)
            cp = math.cos(BASE_PITCH)
            sp = math.sin(BASE_PITCH)
            cr = math.cos(BASE_ROLL)
            sr = math.sin(BASE_ROLL)

            R = np.array([
                [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
                [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
                [-sp, cp*sr, cp*cr]
            ])
            p_world = np.array([x, y, z])
            p_base = R.T @ (p_world - t)

        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.BOX
        primitive.dimensions = [0.05, 0.05, 0.15]

        pose = PoseStamped()
        pose.header = obj.header
        pose.pose.position.x = float(p_base[0])
        pose.pose.position.y = float(p_base[1])
        pose.pose.position.z = float(p_base[2] + 0.075)
        pose.pose.orientation.w = 1.0

        obj.primitives.append(primitive)
        obj.primitive_poses.append(pose.pose)
        obj.operation = CollisionObject.ADD

        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects.append(obj)

        req = ApplyPlanningScene.Request(scene=scene)
        future = client.call_async(req)
        self.executor.spin_until_future_complete(future)

        result = future.result()
        if result and result.success:
            self.node.get_logger().info(
                f"📦 MoveIt collision added in link_base at ({p_base[0]:.3f}, {p_base[1]:.3f}, {p_base[2]+0.075:.3f}) from world [manual={not USE_TF}]"
            )
        else:
            self.node.get_logger().error(f"❌ Failed to add collision object: {name}")



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
        objects = spawner.spawn_random_objects(count=3)
        for obj in objects:
            print(f"已生成: {obj['name']} model={obj['model']} at "
                  f"(world)({obj['pose'].position.x:.2f},{obj['pose'].position.y:.2f},{obj['pose'].position.z:.2f})")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        print("🛑 測試結束")

