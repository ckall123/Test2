# contact_monitor_v2.py
# ------------------------------------------------------
# ContactMonitor with candidate() + stability filtering
# For use in RL-VLM-F pipelines with attach gating logic
# ------------------------------------------------------

import time
from threading import Lock
from gazebo_msgs.msg import ContactsState
from rclpy.node import Node


class FingerContact:
    def __init__(self, finger_name: str):
        self.finger_name = finger_name
        self.contacts: list[str] = []
        self.max_depth: float = 0.0
        self.has_contact: bool = False
        self.last_update_time: float = 0.0
        self.lock = Lock()

    def update(self, msg: ContactsState):
        with self.lock:
            self.contacts.clear()
            self.has_contact = False
            self.max_depth = 0.0
            self.last_update_time = time.time()

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
            self.last_update_time = 0.0


class ContactMonitor:
    def __init__(self, node: Node):
        self.node = node
        self.left_finger = FingerContact("left_finger")
        self.right_finger = FingerContact("right_finger")

        self.node.create_subscription(ContactsState, "/left_finger_contact/contacts", self._left_cb, 10)
        self.node.create_subscription(ContactsState, "/right_finger_contact/contacts", self._right_cb, 10)

        # Cache for temporal candidate tracking
        self._last_candidate: str | None = None
        self._contact_history: list[str] = []

    def _left_cb(self, msg: ContactsState):
        self.left_finger.update(msg)

    def _right_cb(self, msg: ContactsState):
        self.right_finger.update(msg)

    def reset(self):
        self.left_finger.reset()
        self.right_finger.reset()
        self._last_candidate = None
        self._contact_history.clear()

    def check_dual_contact(self, keywords: list[str]) -> tuple[bool, str | None]:
        left_hits = [obj for obj in self.left_finger.contacts if any(k in obj for k in keywords)]
        right_hits = [obj for obj in self.right_finger.contacts if any(k in obj for k in keywords)]
        shared = list(set(left_hits) & set(right_hits))
        both_touching = self.left_finger.has_contact and self.right_finger.has_contact
        return (True, shared[0]) if shared and both_touching else (False, None)

    def candidate(self, keywords: list[str]) -> str | None:
        ok, obj = self.check_dual_contact(keywords)
        if not ok:
            return None
        self._last_candidate = obj
        self._contact_history.append(obj)
        if len(self._contact_history) > 20:
            self._contact_history.pop(0)
        return obj

    def is_stable(self, obj: str, frames: int = 3, v_thresh: float = 0.05, envelope: tuple[float, float] = (0.06, 0.04)) -> bool:
        if not self._contact_history:
            return False
        recent = self._contact_history[-frames:]
        freq = sum(1 for x in recent if x == obj)
        return freq >= frames

    def get_contact_status_string(self) -> str:
        def summarize(contacts: list[str]) -> str:
            return str(contacts[:3]) + ("..." if len(contacts) > 3 else "")

        dual = self.left_finger.has_contact and self.right_finger.has_contact
        return (
            f"[Gripper Contact]\n"
            f"  Left : {self.left_finger.has_contact} (depth={self.left_finger.max_depth:.4f}) \u2192 {summarize(self.left_finger.contacts)}\n"
            f"  Right: {self.right_finger.has_contact} (depth={self.right_finger.max_depth:.4f}) \u2192 {summarize(self.right_finger.contacts)}\n"
            f"  Stable: {self._last_candidate if self.is_stable(self._last_candidate or '') else 'None'}\n"
            f"\u2192 Dual Contact: {dual}"
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

            # 輸出接觸狀態文字
            print(monitor.get_contact_status_string())

            # 嘗試抓候選物件
            obj = monitor.candidate(["object", "beer", "cube", "test"])
            if obj:
                print(f"🎯 候選物件: {obj}")
                stable = monitor.is_stable(obj)
                print(f"  ↳ 穩定接觸: {'✅ YES' if stable else '❌ NO'}")
            else:
                print("🔍 尚無候選物件")

            time.sleep(1.0)
    finally:
        node.destroy_node()
        rclpy.shutdown()
