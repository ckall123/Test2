import random
import numpy as np
import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SpawnEntity
from sim_object import SimObject

X_RANGE = [0.4, 0.7]
Y_RANGE = [-0.2, 0.2]
Z_HEIGHT = 0.05

OBJECT_NAMES = ["cube1", "cylinder1", "sphere1"]

class Spawner(Node):
    """
    Gazebo 物體生成服務節點，用於從模型資料夾中 spawn 物件。
    """
    def __init__(self):
        super().__init__('spawner_node')
        self.cli = self.create_client(SpawnEntity, '/spawn_entity')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /spawn_entity service...')

    def spawn_object(self, name: str, position: np.ndarray):
        req = SpawnEntity.Request()
        req.name = name
        req.xml = f'<sdf version="1.6"><include><uri>model://{name}</uri></include></sdf>'
        req.robot_namespace = name
        req.initial_pose.position.x = float(position[0])
        req.initial_pose.position.y = float(position[1])
        req.initial_pose.position.z = float(position[2])
        req.initial_pose.orientation.w = 1.0  # 無旋轉

        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        if future.result() is not None:
            self.get_logger().info(f'Successfully spawned {name}')
        else:
            self.get_logger().error(f'Failed to spawn {name}: {future.exception()}')


def randomize_objects(num_objects: int = 2) -> list:
    """
    隨機產生物體，呼叫 Gazebo spawn 並設定位置，回傳 SimObject list
    """
    spawner = Spawner()

    objects = []
    selected_names = random.sample(OBJECT_NAMES, num_objects)
    for i, name in enumerate(selected_names):
        x = random.uniform(*X_RANGE)
        y = random.uniform(*Y_RANGE)
        z = Z_HEIGHT
        pos = np.array([x, y, z], dtype=np.float32)
        obj = SimObject(name=name, position=pos)
        spawner.spawn_object(name, pos)
        objects.append(obj)

    spawner.destroy_node()
    return objects