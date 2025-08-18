import random
import numpy as np
import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SpawnEntity
from sim_object import SimObject

X_RANGE = [0.4, 0.7]
Y_RANGE = [-0.2, 0.2]
Z_HEIGHT = 0.05

# OBJECT_NAMES = ["LACING_SHEEP", "Great_Dinos_Triceratops_Toy", "CHICKEN_NESTING", "beer_copy"]
OBJECT_NAMES = ["beer_copy"]

class Spawner(Node):
    def __init__(self):
        super().__init__('spawner_node')
        self.cli = self.create_client(SpawnEntity, '/spawn_entity')
        retries = 0
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /spawn_entity service...')
            retries += 1
            if retries > 10:
                self.get_logger().error('spawn_entity service not available, giving up for now.')
                break

    def spawn_object(self, name: str, position: np.ndarray):
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

        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=20.0)

        # A) 有沒有超時 / 例外
        if not future.done():
            self.get_logger().error("Spawn timed out after 20s")
            return
        exc = future.exception()
        if exc:
            self.get_logger().error(f"Spawn call raised: {exc!r}")
            return

        # B) 真的結果（關鍵在這裡）
        res = future.result()
        if res.success:
            self.get_logger().info(f"Spawned {name}: {res.status_message}")
        else:
            self.get_logger().error(f"Failed to spawn {name}: {res.status_message}")


def randomize_objects(num_objects: int = 2) -> list:
    should_shutdown = False
    if not rclpy.ok():
        rclpy.init()
        should_shutdown = True
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
    if should_shutdown and rclpy.ok():
        rclpy.shutdown()
    return objects