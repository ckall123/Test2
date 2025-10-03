#!/usr/bin/env python3
# ik_min.py
import rclpy
from rclpy.node import Node
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import RobotState
from moveit_msgs.srv import GetPositionIK
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener, LookupException
import math

ARM_ORDER = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]

def solve_ik(pos_xyz, quat_xyzw, group="xarm6", base="link_base", tip="link6",
             seed_wait=5.0, timeout=2.0,
             joint_states_topic="/joint_states", service_name="/compute_ik"):
    need_init = not rclpy.ok()
    if need_init:
        rclpy.init()
    node = Node("ik_solver")

    seed = {"msg": None}
    def on_js(msg):
        m = {n: p for n, p in zip(msg.name, msg.position)}
        if all(k in m for k in ARM_ORDER):
            seed["msg"] = JointState(name=ARM_ORDER, position=[m[k] for k in ARM_ORDER])
    node.create_subscription(JointState, joint_states_topic, on_js, 10)

    t0 = node.get_clock().now().nanoseconds / 1e9
    while seed["msg"] is None and node.get_clock().now().nanoseconds / 1e9 - t0 < seed_wait:
        rclpy.spin_once(node, timeout_sec=0.1)
    seed_msg = seed["msg"] or JointState(name=ARM_ORDER, position=[0.0]*6)

    cli = node.create_client(GetPositionIK, service_name)
    cli.wait_for_service(timeout_sec=seed_wait)

    ps = PoseStamped()
    ps.header.frame_id = base
    ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = pos_xyz
    ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z, ps.pose.orientation.w = quat_xyzw

    req = GetPositionIK.Request()
    req.ik_request.group_name = group
    req.ik_request.pose_stamped = ps
    req.ik_request.avoid_collisions = False
    req.ik_request.timeout = Duration(sec=int(timeout), nanosec=int((timeout % 1)*1e9))
    req.ik_request.ik_link_name = tip
    req.ik_request.robot_state = RobotState(joint_state=seed_msg)

    fut = cli.call_async(req)
    rclpy.spin_until_future_complete(node, fut, timeout_sec=timeout + 1.0)
    res = fut.result()

    m = {n: p for n, p in zip(res.solution.joint_state.name, res.solution.joint_state.position)} if res else {}
    sol6 = [m.get(k, 0.0) for k in ARM_ORDER]
    code = res.error_code.val if res else -999

    node.destroy_node()
    if need_init:
        rclpy.shutdown()
    return code, sol6, seed_msg.position

def get_current_pose(node: Node, base="link_base", tip="link6"):
    tf_buffer = Buffer()
    TransformListener(tf_buffer, node)
    for _ in range(30):
        try:
            t = tf_buffer.lookup_transform(base, tip, rclpy.time.Time(), rclpy.duration.Duration(seconds=0.1))
            tr, rot = t.transform.translation, t.transform.rotation
            return (tr.x, tr.y, tr.z), (rot.x, rot.y, rot.z, rot.w)
        except LookupException:
            rclpy.spin_once(node, timeout_sec=0.1)
    node.get_logger().error("TF lookup failed after retries")
    return None, None

if __name__ == "__main__":
    rclpy.init()
    node = Node("ik_debug_node")
    pos, quat = get_current_pose(node)
    if pos:
        code, sol, real = solve_ik(pos, quat)
        to_deg = lambda v: [round(math.degrees(x), 2) for x in v]
        sol_deg, real_deg = to_deg(sol), to_deg(real)
        diff = [round(s - r, 2) for s, r in zip(sol_deg, real_deg)]
        print("code:", code)
        print("real joint (deg):", real_deg)
        print("sol6  joint (deg):", sol_deg)
        print("diff (deg):", diff)
    else:
        print("無法取得 TF，無法執行 IK")
    node.destroy_node()
    rclpy.shutdown()
