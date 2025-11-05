import subprocess
from typing import Set, Dict  # 新增 Dict

ATTACH_SERVICE = "/ATTACHLINK"
DETACH_SERVICE = "/DETACHLINK"
ATTACH_TYPE = "linkattacher_msgs/srv/AttachLink"
DETACH_TYPE = "linkattacher_msgs/srv/DetachLink"
DEFAULT_OBJECT_LINK = "link"

class AttachDetachClient:
    def __init__(self, node, model_name="UF_ROBOT", finger_link="right_finger"):
        self.node = node
        self.model_name = model_name
        self.finger_link = finger_link

        self.attached_set: Set[str] = set()
        self._finger_used: Dict[str, str] = {}  # 🔧 key = "model::link" → 記錄當時用哪根手指

    def _call_service(self, service: str, args: str, srv_type: str) -> bool:
        cmd = ["ros2", "service", "call", service, srv_type, args]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=6.0)  # 拉長 timeout
            if result.returncode == 0:
                self.node.get_logger().info(f"[Attacher] ✅ Service success: {service}")
                return True
            else:
                self.node.get_logger().warn(f"[Attacher] ❌ Service failed: {result.stderr.strip()}")
                return False
        except Exception as e:
            self.node.get_logger().warn(f"[Attacher] ❌ Exception: {e}")
            return False

    def attach(self, model: str, link: str, finger: str | None = None) -> bool:
        fid = finger or self.finger_link
        key = f"{model}::{link}"
        if key in self.attached_set:
            self.node.get_logger().info(f"[Attacher] 🔁 Already attached: {key}")
            return True

        args = (
            f"{{model1_name: '{self.model_name}', link1_name: '{fid}', "
            f"model2_name: '{model}', link2_name: '{link}'}}"
        )
        ok = self._call_service(ATTACH_SERVICE, args, ATTACH_TYPE)
        if ok:
            self.attached_set.add(key)
            self._finger_used[key] = fid  # ✅ 記住是哪根手指 attach 的
        return ok

    def detach(self, model: str, link: str) -> bool:
        key = f"{model}::{link}"
        if key not in self.attached_set:
            self.node.get_logger().info(f"[Attacher] ℹ️ Not attached: {key}")
            return True

        fid = self._finger_used.get(key, self.finger_link)  # ✅ 優先用當時 attach 時的手指
        args = (
            f"{{model1_name: '{self.model_name}', link1_name: '{fid}', "
            f"model2_name: '{model}', link2_name: '{link}'}}"
        )
        ok = self._call_service(DETACH_SERVICE, args, DETACH_TYPE)
        if ok:
            self.attached_set.remove(key)
            self._finger_used.pop(key, None)
        return ok

    def is_attached(self, model: str, link: str) -> bool:
        return f"{model}::{link}" in self.attached_set

    def clear_all(self):
        self.node.get_logger().info("[Attacher] 🧹 Clearing all attachment records")
        self.attached_set.clear()
        self._finger_used.clear()  # ✅ 一起清空


# 🧪 測試入口
if __name__ == '__main__':
    import rclpy
    from rclpy.executors import SingleThreadedExecutor

    rclpy.init()
    node = rclpy.create_node("attacher_test")
    executor = SingleThreadedExecutor()
    executor.add_node(node)

    attacher = AttachDetachClient(node)
    model = "wood_cube_5cm"
    link = DEFAULT_OBJECT_LINK

    try:
        print("\n🔗 Attach...")
        if attacher.attach(model, link):
            print("✅ Attach success")

        print("\n🔎 Check...")
        print(f"Is attached? {'🟢 YES' if attacher.is_attached(model, link) else '🔴 NO'}")

        print("\n❌ Detach...")
        if attacher.detach(model, link):
            print("✅ Detach success")

        print(f"After detach? {'🟢 YES' if attacher.is_attached(model, link) else '🔴 NO'}")
    finally:
        node.destroy_node()
        rclpy.shutdown()