"""
objects/xarm_positions.py — minimal & clean

這個版本只保留最必要的功能：
- warm_tf(): 幫忙暖機 TF buffer
- get_link_xyz(): 拿單一 link 在某個 reference 下的 (x,y,z)
- get_arm_bounds(): 計算手臂的 AABB
- get_gripper_span(): 簡潔版的夾爪座標取得（若不指定 reference，會嘗試少數常用 reference 並回傳第一個成功的結果）

移除了：
- 先前那個會回傳大量診斷訊息的 auto 函式（get_gripper_span_auto）
- 冗長的診斷/例外追蹤字串

設計原則：失敗就丟例外（呼叫端自行 try/except），函數單一職責、易讀易維護。
"""
from typing import Dict, Tuple, List, Optional
import rclpy
from rclpy.time import Time
from rclpy.duration import Duration
from tf2_ros import Buffer

# 可自訂的 link 清單
LINK_NAMES: List[str] = [
    "link_base", "link1", "link2", "link3", "link4", "link5", "link6",
    "left_finger", "right_finger",
]

"""
reference_frame:
    "link_base"      手臂的底座（預設值）                          最常用的基準點，用來看整隻手臂
    "world"          真正的全域原點（通常固定在地板或地圖中）        如果你要對應到 Gazebo、地圖、全局定位等
    "link_eef"       手臂最末端的端點                              通常用來看夾爪相對於末端的相對關係
    "camera_frame"   如果有攝影機、深度感測器之類的東西             用來看感測器看到的目標位置
"""

# 若不給 reference，get_gripper_span 會依序嘗試的參考 frame（短清單，避免花太多時間）
TRY_REFERENCE_FRAMES: List[str] = ["world", "link_eef", "link_base"]

# -----------------------
# TF helpers
# -----------------------
def warm_tf(node: rclpy.node.Node, tf_buffer: Buffer, spins: int = 40, timeout_sec: float = 0.02) -> None:
    """讓 TransformListener / tf_buffer 暖機：多次 spin_once 接收 /tf 更新。"""
    for _ in range(spins):
        rclpy.spin_once(node, timeout_sec=timeout_sec)


def get_link_xyz(tf_buffer: Buffer, target: str, reference_frame: str = "world", timeout_sec: float = 1.0) -> Tuple[float, float, float]:
    """取得 target 在 reference_frame 底下的 (x,y,z)。若查詢失敗會丟出 tf2 的 Exception。"""
    transform = tf_buffer.lookup_transform(reference_frame, target, Time(), timeout=Duration(seconds=timeout_sec))
    tr = transform.transform.translation
    return float(tr.x), float(tr.y), float(tr.z)


def get_arm_bounds(tf_buffer: Buffer, links: List[str] = LINK_NAMES, margin: float = 0.02, reference_frame: str = "world") -> Dict[str, Tuple[float, float]]:
    """以 AABB 回傳手臂在指定 reference 下的範圍 (x,y,z)。會略過抓不到的 link，若全部失敗則丟 RuntimeError。"""
    poses = []
    for lk in links:
        try:
            poses.append(get_link_xyz(tf_buffer, lk, reference_frame))
        except Exception:
            continue
    if not poses:
        raise RuntimeError("No TF poses available for provided links")
    xs = [p[0] for p in poses]
    ys = [p[1] for p in poses]
    zs = [p[2] for p in poses]
    return {
        "x": (min(xs) - margin, max(xs) + margin),
        "y": (min(ys) - margin, max(ys) + margin),
        "z": (min(zs) - margin, max(zs) + margin),
    }


def get_gripper_span(tf_buffer: Buffer, reference_frame: Optional[str] = None, timeout_sec: float = 1.0) -> Dict[str, object]:
    """取得夾爪在某個 reference 下的 span。若 reference_frame 為 None，會嘗試 TRY_REFERENCE_FRAMES 中的 frame，並回傳第一個成功的結果。

    回傳 dict: { 'reference_used': str, 'y_left': float, 'y_right': float, 'width': float, 'x_center': float, 'z_center': float }
    若全部嘗試仍失敗，會丟 RuntimeError。
    """
    refs = [reference_frame] if reference_frame is not None else TRY_REFERENCE_FRAMES
    for ref in refs:
        if ref is None:
            continue
        try:
            lx, ly, lz = get_link_xyz(tf_buffer, "left_finger", reference_frame=ref, timeout_sec=timeout_sec)
            rx, ry, rz = get_link_xyz(tf_buffer, "right_finger", reference_frame=ref, timeout_sec=timeout_sec)
            return {
                "reference_used": ref,
                "y_left": ly,
                "y_right": ry,
                "width": abs(ry - ly),
                "x_center": (lx + rx) / 2.0,
                "y_center": (ly + ry) / 2.0,
                "z_center": (lz + rz) / 2.0,
            }
        except Exception:
            continue
    raise RuntimeError("Unable to get gripper span using available reference frames")
