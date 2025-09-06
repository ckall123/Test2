# ik_with_seed.py
import rclpy, time
from rclpy.node import Node
from sensor_msgs.msg import JointState
from moveit_msgs.srv import GetPositionIK
from geometry_msgs.msg import PoseStamped
from builtin_interfaces.msg import Duration

GROUP = "xarm6"
TIP   = "link_tcp"

POS = (0.207, 0.0, -0.060)         # 你 tf2_echo 的位置
QUAT= (1.0, 0.0, 0.0, 0.0)         # 你 tf2_echo 的 (x,y,z,w)

class IK(Node):
    def __init__(self):
        super().__init__("ik_with_seed")
        self.seed = None
        self.create_subscription(JointState, "/joint_states", self._cb, 10)
        self.cli = self.create_client(GetPositionIK, "/compute_ik")
        self.cli.wait_for_service(10.0)

    def _cb(self, msg):
        if self.seed is None and msg.name:
            self.seed = msg

    def solve(self):
        # 等到讀到一次 joint_states
        while rclpy.ok() and self.seed is None:
            rclpy.spin_once(self, timeout_sec=0.1)
        req = GetPositionIK.Request()
        req.ik_request.group_name   = GROUP
        req.ik_request.ik_link_name = TIP
        req.ik_request.robot_state.joint_state = self.seed  # ★ 用當前關節當 seed
        ps = PoseStamped()
        ps.header.frame_id = "link_base"
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = POS
        ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z, ps.pose.orientation.w = QUAT
        req.ik_request.pose_stamped = ps
        req.ik_request.timeout = Duration(sec=1)
        req.ik_request.avoid_collisions = False
        fut = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut)
        return fut.result()

def main():
    rclpy.init()
    node = IK()
    time.sleep(1.0)  # 給 TF/參數時間 ready
    res = node.solve()
    print("error_code:", res.error_code.val)  # 1=SUCCESS
    print("names:", list(res.solution.joint_state.name))
    print("pos  :", list(res.solution.joint_state.position))
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
