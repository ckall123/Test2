#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from geometry_msgs.msg import Pose
from shape_msgs.msg import SolidPrimitive
from moveit_msgs.msg import CollisionObject, PlanningScene
from moveit_msgs.srv import ApplyPlanningScene

class AddStaticTable(Node):
    def __init__(self):
        super().__init__('add_static_table')
        self.cli = self.create_client(ApplyPlanningScene, '/apply_planning_scene')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('等待 /apply_planning_scene 服務中...')
        self.add_table()

    def add_table(self):
        table = CollisionObject()
        table.id = 'gazebo_table'
        table.header = Header(frame_id='world')   # ← 改成你 MoveIt 的世界座標系

        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [0.6, 1.2, 0.75]  # X, Y, Z (m)

        pose = Pose()
        # pose.position.x = -0.3
        # pose.position.y = -0.8
        pose.position.x = -0.2
        pose.position.y = 0.1
        pose.position.z = -0.38 # 以中心為原點，所以是高度一半
        pose.orientation.w = 1.0

        table.primitives.append(box)
        table.primitive_poses.append(pose)
        table.operation = CollisionObject.ADD

        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects.append(table)

        req = ApplyPlanningScene.Request(scene=scene)
        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        ok = bool(future.result() and future.result().success)
        self.get_logger().info(f'套用場景：{"成功" if ok else "失敗"}')
        rclpy.shutdown()

def main():
    rclpy.init()
    AddStaticTable()

if __name__ == '__main__':
    main()
