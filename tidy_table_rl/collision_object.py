from std_msgs.msg import Header
from geometry_msgs.msg import Pose
from shape_msgs.msg import SolidPrimitive
from moveit_msgs.msg import CollisionObject, PlanningScene
from moveit_msgs.srv import ApplyPlanningScene

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node


class CollisionObjectManager:
    def __init__(self, node: Node, executor: SingleThreadedExecutor):
        self.node = node
        self.executor = executor
        self.cli = self.node.create_client(ApplyPlanningScene, '/apply_planning_scene')

        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info('等待 /apply_planning_scene 服務中...')

        self.objects = []

    def _apply_scene(self):
        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects = self.objects
        req = ApplyPlanningScene.Request(scene=scene)
        future = self.cli.call_async(req)
        self.executor.spin_until_future_complete(future)
        res = future.result()
        if not res or not res.success:
            self.node.get_logger().warn("❌ ApplyPlanningScene failed")

    def add_table(self, name: str, pose: Pose, size=(0.6, 1.2, 0.75)):
        self._add_box(name, pose, size)

    def add_box(self, name: str, pose: Pose, size=(0.1, 0.1, 0.1)):
        self._add_box(name, pose, size)

    def _add_box(self, name: str, pose: Pose, size):
        self.remove(name)

        obj = CollisionObject()
        obj.id = name
        obj.header = Header(frame_id='world')

        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = list(size)

        obj.primitives.append(box)
        obj.primitive_poses.append(pose)
        obj.operation = CollisionObject.ADD

        self.objects.append(obj)
        self._apply_scene()

    def remove(self, name: str):
        self.objects = [o for o in self.objects if o.id != name]

        obj = CollisionObject()
        obj.id = name
        obj.header = Header(frame_id='world')
        obj.operation = CollisionObject.REMOVE

        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects.append(obj)

        req = ApplyPlanningScene.Request(scene=scene)
        future = self.cli.call_async(req)
        self.executor.spin_until_future_complete(future)
        res = future.result()
        if not res or not res.success:
            self.node.get_logger().warn(f"⚠️ Timeout removing {name}")

    def clear(self):
        for obj in list(self.objects):
            self.remove(obj.id)
        self.objects.clear()

    @classmethod
    def default_setup(cls, node: Node, executor: SingleThreadedExecutor) -> "CollisionObjectManager":
        manager = cls(node, executor)

        pose = Pose()
        pose.position.x = -0.2
        pose.position.y = 0.1
        pose.position.z = -0.38
        pose.orientation.w = 1.0

        manager.add_table("table", pose)
        return manager

    def shutdown(self):
        self.node.destroy_node()


if __name__ == '__main__':
    rclpy.init()
    executor = SingleThreadedExecutor()
    node = rclpy.create_node("collision_object_node")
    executor.add_node(node)

    manager = CollisionObjectManager.default_setup(node, executor)
    node.get_logger().info("✅ CollisionObjectManager ready.")

    input("🛠️ Enter 鍵以清除並結束...")
    manager.clear()
    manager.shutdown()
    rclpy.shutdown()
