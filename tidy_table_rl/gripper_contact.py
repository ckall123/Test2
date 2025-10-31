import time
from threading import Lock
from gazebo_msgs.msg import ContactsState
from rclpy.node import Node


class FingerContact:
    """儲存單側手指的接觸資訊。"""
    def __init__(self, finger_name: str):
        self.finger_name = finger_name
        self.contacts: list[str] = []
        self.max_depth: float = 0.0
        self.has_contact: bool = False
        self.lock = Lock()

    def update(self, msg: ContactsState):
        with self.lock:
            self.contacts.clear()
            self.has_contact = False
            self.max_depth = 0.0

            contact_set = {
                state.collision2_name if self.finger_name in state.collision1_name else state.collision1_name
                for state in msg.states
            }
            self.contacts.extend(contact_set)

            self.max_depth = max((max(state.depths) for state in msg.states if state.depths), default=0.0)
            self.has_contact = any(d > 2e-4 for state in msg.states for d in state.depths)

    def reset(self):
        with self.lock:
            self.contacts.clear()
            self.has_contact = False
            self.max_depth = 0.0


class ContactMonitor:
    """監控雙指接觸狀態，提供查詢與重置介面。"""
    def __init__(self, node: Node):
        self.node = node
        self.left_finger = FingerContact('left_finger')
        self.right_finger = FingerContact('right_finger')

        self.node.create_subscription(
            ContactsState, '/left_finger_contact/contacts', self._left_cb, 10)
        self.node.create_subscription(
            ContactsState, '/right_finger_contact/contacts', self._right_cb, 10)

    def _left_cb(self, msg: ContactsState):
        self.left_finger.update(msg)

    def _right_cb(self, msg: ContactsState):
        self.right_finger.update(msg)

    def reset(self):
        """清除所有接觸狀態（用於每回合開頭）"""
        self.left_finger.reset()
        self.right_finger.reset()

    def check_dual_contact(self, keywords: list[str]) -> tuple[bool, str | None]:
        """是否雙指皆接觸相同目標（支援關鍵字模糊匹配）"""
        left_hits = [obj for obj in self.left_finger.contacts if any(k in obj for k in keywords)]
        right_hits = [obj for obj in self.right_finger.contacts if any(k in obj for k in keywords)]
        shared = list(set(left_hits) & set(right_hits))
        both_touching = self.left_finger.has_contact and self.right_finger.has_contact

        return (True, shared[0]) if shared and both_touching else (False, None)

    def in_contact(self, target: str | None = None) -> str | None:
        """是否雙指接觸指定目標（或任意共同接觸物）"""
        if not (self.left_finger.has_contact and self.right_finger.has_contact):
            return None
        if target is None:
            shared = set(self.left_finger.contacts) & set(self.right_finger.contacts)
            return next(iter(shared), None)
        ok, obj = self.check_dual_contact([target])
        return obj if ok else None

    def get_contact_status_string(self) -> str:
        """美化後的接觸狀態輸出（for log）"""
        def summarize(contacts: list[str]) -> str:
            return str(contacts[:3]) + ("..." if len(contacts) > 3 else "")

        dual = self.left_finger.has_contact and self.right_finger.has_contact
        return (
            f"[Gripper Contact]\n"
            f"  Left : {self.left_finger.has_contact} (depth={self.left_finger.max_depth:.4f}) → {summarize(self.left_finger.contacts)}\n"
            f"  Right: {self.right_finger.has_contact} (depth={self.right_finger.max_depth:.4f}) → {summarize(self.right_finger.contacts)}\n"
            f"→ Dual Contact: {dual}"
        )


# --- 測試入口 ---
if __name__ == '__main__':
    import rclpy
    from rclpy.executors import SingleThreadedExecutor

    rclpy.init()
    node = rclpy.create_node("contact_monitor_test")
    executor = SingleThreadedExecutor()
    executor.add_node(node)

    monitor = ContactMonitor(node)

    try:
        while rclpy.ok():
            executor.spin_once(timeout_sec=0.1)
            print(monitor.get_contact_status_string())

            ok, obj = monitor.check_dual_contact(["test", "beer"])
            print(f"✅ 同時接觸：{obj}" if ok else "⏳ 尚未接觸相同物件")
            time.sleep(1)
    finally:
        node.destroy_node()
        rclpy.shutdown()
