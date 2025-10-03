import time
from threading import Lock

from gazebo_msgs.msg import ContactsState
from rclpy.node import Node


class FingerContact:
    """儲存單側手指的接觸資訊。"""
    def __init__(self, finger_name: str):
        self.finger_name = finger_name
        self.contacts = []
        self.max_depth = 0.0
        self.has_contact = False
        self.lock = Lock()

    def update(self, msg: ContactsState):
        with self.lock:
            contact_set = set()
            self.has_contact = False
            self.max_depth = 0.0

            for state in msg.states:
                link1, link2 = state.collision1_name, state.collision2_name
                other = link2 if self.finger_name in link1 else link1
                contact_set.add(other)

                if state.depths:
                    depth = max(state.depths)
                    self.max_depth = max(self.max_depth, depth)
                    if depth > 2e-4:
                        self.has_contact = True

            self.contacts = list(contact_set)


class ContactMonitor:
    """監控雙指接觸狀態，提供查詢函式與完整狀態文字輸出。"""
    def __init__(self, node: Node):
        self.node = node
        self.left_finger = FingerContact('left_finger')
        self.right_finger = FingerContact('right_finger')

        self.node.create_subscription(
            ContactsState, '/left_finger_contact/contacts', self._left_cb, 10)
        self.node.create_subscription(
            ContactsState, '/right_finger_contact/contacts', self._right_cb, 10)

    def _left_cb(self, msg):
        self.left_finger.update(msg)

    def _right_cb(self, msg):
        self.right_finger.update(msg)

    def check_dual_contact(self, keywords: list[str]):
        """檢查左右是否都接觸到相同關鍵字物件，回傳 (bool, 物件名 or None)"""
        left_hits = [obj for obj in self.left_finger.contacts if any(k in obj for k in keywords)]
        right_hits = [obj for obj in self.right_finger.contacts if any(k in obj for k in keywords)]

        shared = list(set(left_hits) & set(right_hits))
        if shared and self.left_finger.has_contact and self.right_finger.has_contact:
            return True, shared[0]
        return False, None

    def get_contact_status_string(self):
        """取得漂亮的 log-style 狀態文字。"""
        def summarize(contacts):
            if not contacts:
                return "[]"
            display = contacts[:3]
            more = len(contacts) - len(display)
            return f"{display}... (+{more})" if more > 0 else str(display)

        dual = self.left_finger.has_contact and self.right_finger.has_contact

        return (
            f"[Gripper Contact]\n"
            f"  Left : {self.left_finger.has_contact} (depth={self.left_finger.max_depth:.4f}) → {summarize(self.left_finger.contacts)}\n"
            f"  Right: {self.right_finger.has_contact} (depth={self.right_finger.max_depth:.4f}) → {summarize(self.right_finger.contacts)}\n"
            f"→ Dual Contact: {dual}"
        )

    def in_contact(self, target: str | None = None) -> str | None:
        """查詢是否同時接觸目標物件（如有）"""
        if not (self.left_finger.has_contact and self.right_finger.has_contact):
            return None

        if target is None:
            shared = set(self.left_finger.contacts) & set(self.right_finger.contacts)
            return next(iter(shared)) if shared else None

        ok, obj = self.check_dual_contact([target])
        return obj if ok else None


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

            ok, obj = monitor.check_dual_contact(["test", "beer"])
            print(monitor.get_contact_status_string())
            if ok:
                print(f"✅ 同時夾住了：{obj}")
            else:
                print("⏳ 還沒同時接觸同一個物件")
            print("------")
            time.sleep(1)

    finally:
        node.destroy_node()
        rclpy.shutdown()
