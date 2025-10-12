# control/collision_object.py
from std_msgs.msg import Header
from geometry_msgs.msg import Pose
from shape_msgs.msg import SolidPrimitive
from moveit_msgs.msg import CollisionObject, PlanningScene
from moveit_msgs.srv import ApplyPlanningScene
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node


class CollisionObjectManager:
    def __init__(self, node=None, executor=None):
        if node is None or executor is None:
            rclpy.init()
            self.executor = SingleThreadedExecutor()
            self.node = rclpy.create_node('collision_object_manager')
            self.executor.add_node(self.node)
            self.owns_node = True
        else:
            self.node = node
            self.executor = executor
            self.owns_node = False

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
        result = future.result()
        if result and result.success:
            self.node.get_logger().info('已成功更新 CollisionObject。')
        else:
            self.node.get_logger().error('更新 CollisionObject 失敗。')

    def add_table(self, name: str, pose: Pose, size=(0.6, 1.2, 0.75)):
        self._add_box(name, pose, size)

    def add_box(self, name: str, pose: Pose, size=(0.1, 0.1, 0.1)):
        self._add_box(name, pose, size)

    def _add_box(self, name: str, pose: Pose, size):
        # Remove existing object with same ID if any
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

        self.node.get_logger().info(f"已刪除(可能重名) CollisionObject: {name}")

    def clear(self):
        for obj in self.objects:
            self.remove(obj.id)
        self.objects.clear()

    def shutdown(self):
        if self.owns_node:
            self.node.destroy_node()
            rclpy.shutdown()


# Utility function for training call
collision_object_manager = None

def setup_collision_objects():
    global collision_object_manager
    if collision_object_manager is None:
        collision_object_manager = CollisionObjectManager()

    pose = Pose()
    pose.position.x = -0.2
    pose.position.y = 0.1
    pose.position.z = -0.38
    pose.orientation.w = 1.0

    collision_object_manager.add_table("table", pose, size=(0.6, 1.2, 0.75))


def shutdown_collision_objects():
    if collision_object_manager:
        collision_object_manager.shutdown()


if __name__ == '__main__':
    setup_collision_objects()
    shutdown_collision_objects()
