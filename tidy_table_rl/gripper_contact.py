import rclpy
from rclpy.node import Node
from gazebo_msgs.msg import ContactsState
from threading import Lock, Thread
import time


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


class ContactMonitor(Node):
    """監控雙指接觸狀態，提供查詢函式與完整狀態文字輸出。"""
    _instance = None

    def __init__(self):
        super().__init__('contact_monitor')
        self.left_finger = FingerContact('left_finger')
        self.right_finger = FingerContact('right_finger')

        self.create_subscription(ContactsState, '/left_finger_contact/contacts', self._left_cb, 10)
        self.create_subscription(ContactsState, '/right_finger_contact/contacts', self._right_cb, 10)

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

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            if not rclpy.ok():
                rclpy.init(args=None)
            cls._instance = cls()
            thread = Thread(target=rclpy.spin, args=(cls._instance,), daemon=True)
            thread.start()
        return cls._instance


# --- 對外函式介面（供控制器等模組使用） ---
def in_contact(target: str | None = None) -> str | None:
    """
    查詢左右手指是否同時接觸某個目標。
    - target=None：若同時接觸任一共同物件，回傳該物件名
    - target='xxx'：若同時接觸名字包含 'xxx' 的物件，回傳該物件名
    否則回傳 None
    """
    monitor = ContactMonitor.get_instance()

    if not (monitor.left_finger.has_contact and monitor.right_finger.has_contact):
        return None

    if target is None:
        shared = set(monitor.left_finger.contacts) & set(monitor.right_finger.contacts)
        return next(iter(shared)) if shared else None

    ok, obj = monitor.check_dual_contact([target])
    return obj if ok else None


# --- 測試用程式入口 ---
def main():
    monitor = ContactMonitor.get_instance()

    while True:
        ok, obj = monitor.check_dual_contact(["test", "beer"])
        print(monitor.get_contact_status_string())
        if ok:
            print(f"✅ 同時夾住了：{obj}")
        else:
            print("⏳ 還沒同時接觸同一個物件")
        print("------")
        time.sleep(1)


if __name__ == '__main__':
    main()
