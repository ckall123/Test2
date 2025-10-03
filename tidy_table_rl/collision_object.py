# control/collision_object.py
from std_msgs.msg import Header
from geometry_msgs.msg import Pose, PoseStamped
from shape_msgs.msg import SolidPrimitive
from moveit_msgs.msg import CollisionObject, PlanningScene
from moveit_msgs.srv import ApplyPlanningScene


class CollisionObjectManager:
    def __init__(self, node, executor):
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
        self.executor.spin_until_future_complete(future)  # 這裡修正
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
        obj = CollisionObject()
        obj.id = name
        obj.header = Header(frame_id='world')
        obj.operation = CollisionObject.REMOVE
        self.objects = [o for o in self.objects if o.id != name]

        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects.append(obj)
        req = ApplyPlanningScene.Request(scene=scene)
        future = self.cli.call_async(req)
        self.executor.spin_until_future_complete(future)  # 修正

        self.node.get_logger().info(f"已刪除 CollisionObject: {name}")

    def clear(self):
        for obj in self.objects:
            self.remove(obj.id)
        self.objects.clear()


if __name__ == '__main__':
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from geometry_msgs.msg import Pose

    rclpy.init()
    executor = SingleThreadedExecutor()
    node = Node('collision_object_test')
    executor.add_node(node)

    manager = CollisionObjectManager(node=node, executor=executor)

    pose = Pose()
    pose.position.x = -0.2
    pose.position.y = 0.1
    pose.position.z = -0.38
    pose.orientation.w = 1.0

    manager.add_table("demo_table", pose, size=(0.6, 1.2, 0.75))

    node.destroy_node()
    rclpy.shutdown()
