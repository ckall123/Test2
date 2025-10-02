# sim/gazebo_if.py
import os
import math
import random
from pathlib import Path
import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SpawnEntity, DeleteEntity, SetEntityState
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import GetEntityState
from std_srvs.srv import Empty
from geometry_msgs.msg import Pose

# 桌面範圍
X_RANGE = [-1.05, 0.45]
Y_RANGE = [-1.20, -0.40]
Z_HEIGHT = 1.015

# 機械臂佔用區域
ARM_X = (-0.45, -0.11)
ARM_Y = (-0.67, -0.26)

class GazeboInterface(Node):
    def __init__(self):
        super().__init__('gazebo_interface')
        self.cli_spawn = self.create_client(SpawnEntity, '/spawn_entity')
        self.cli_delete = self.create_client(DeleteEntity, '/delete_entity')
        self.cli_set_state = self.create_client(SetEntityState, '/set_entity_state')
        self.cli_get_state = self.create_client(GetEntityState, '/get_entity_state')
        self.cli_pause = self.create_client(Empty, '/pause_physics')
        self.cli_unpause = self.create_client(Empty, '/unpause_physics')
        self.cli_reset = self.create_client(Empty, '/reset_world')

    def spawn(self, name: str, xml: str, pose: Pose):
        req = SpawnEntity.Request()
        req.name = name
        req.xml = xml
        req.initial_pose = pose
        self.cli_spawn.call_async(req)

    def spawn_model_by_name(self, name: str, model_name: str, pose: Pose):
        model_path = Path.home() / '.gazebo' / 'models' / model_name / 'model.sdf'
        if not model_path.exists():
            self.get_logger().error(f'Model file not found: {model_path}')
            return
        xml_string = model_path.read_text()
        self.spawn(name, xml_string, pose)

    def spawn_multiple_models(self, model_names, count):
        name_counts = {}
        spawned_names = []

        def is_valid(x, y):
            return not (ARM_X[0] <= x <= ARM_X[1] and ARM_Y[0] <= y <= ARM_Y[1])

        for _ in range(count):
            model = random.choice(model_names)
            name_counts[model] = name_counts.get(model, 0) + 1
            suffix = f"_{name_counts[model]}" if name_counts[model] > 1 else ""
            unique_name = f"{model}{suffix}"

            # 隨機位置直到不碰機械臂
            for _ in range(100):  # 最多試100次
                x = random.uniform(X_RANGE[0], X_RANGE[1])
                y = random.uniform(Y_RANGE[0], Y_RANGE[1])
                if is_valid(x, y):
                    break
            else:
                self.get_logger().warn(f"Failed to find valid position for {unique_name}, skipping.")
                continue

            pose = Pose()
            pose.position.x = x
            pose.position.y = y
            pose.position.z = Z_HEIGHT

            self.spawn_model_by_name(unique_name, model, pose)
            spawned_names.append(unique_name)
            self.get_logger().info(f"Spawned {unique_name} at ({x:.2f}, {y:.2f})")

        return spawned_names

    def delete(self, name: str):
        req = DeleteEntity.Request()
        req.name = name
        self.cli_delete.call_async(req)

    def set_model_state(self, name: str, pose: Pose):
        state = ModelState()
        state.name = name
        state.pose = pose
        req = SetEntityState.Request()
        req.state = state
        self.cli_set_state.call_async(req)

    def get_model_state(self, name: str):
        req = GetEntityState.Request()
        req.name = name
        req.reference_frame = 'world'
        future = self.cli_get_state.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        return future.result().state.pose if future.result() else None

    def pause(self):
        self.cli_pause.call_async(Empty.Request())

    def unpause(self):
        self.cli_unpause.call_async(Empty.Request())

    def reset_world(self):
        self.cli_reset.call_async(Empty.Request())


def main():
    rclpy.init()
    node = GazeboInterface()
    node.get_logger().info("Gazebo interface node started.")

    # Demo: spawn 5 random models from list
    spawned = node.spawn_multiple_models(
        model_names=["wood_cube_5cm", "wood_cube_10cm"],
        count=5
    )
    rclpy.spin_once(node, timeout_sec=2.0)

    for name in spawned:
        # node.delete(name)
        node.get_logger().info(f"Deleted {name}")

    rclpy.shutdown()


if __name__ == '__main__':
    main()