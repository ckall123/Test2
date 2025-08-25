# rl_utils/contact_detection.py
import numpy as np
from geometry_msgs.msg import Pose


def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array([p1.x, p1.y, p1.z]) - np.array([p2.x, p2.y, p2.z]))


def is_in_contact(gripper_pose: Pose, object_pose: Pose, threshold: float = 0.05) -> bool:
    """
    判斷 gripper 跟物體的距離是否小於 threshold，即認定為碰觸
    """
    return euclidean_distance(gripper_pose.position, object_pose.position) < threshold


def any_object_in_contact(gripper_pose: Pose, objects: list[Pose], threshold: float = 0.05) -> bool:
    """
    判斷 gripper 是否有碰到任一個物體
    """
    return any(is_in_contact(gripper_pose, obj_pose, threshold) for obj_pose in objects)


def objects_in_contact(gripper_pose: Pose, objects: dict[str, Pose], threshold: float = 0.05) -> list[str]:
    """
    回傳與 gripper 接觸的所有物件名稱
    """
    return [name for name, pose in objects.items() if is_in_contact(gripper_pose, pose, threshold)]


# === 桌子區域碰撞偵測 ===

TABLE_BOUNDS = {
    "x": (-0.75, 0.75),   # 桌子 x 範圍: 中心 0，寬 1.5m
    "y": (-0.4, 0.4),     # 桌子 y 範圍: 中心 0，深度 0.8m
    "z": (0.985, 1.015),  # 桌面厚度 ±0.015 around z=1.0m
}


def is_touching_table(gripper_pose: Pose) -> bool:
    """
    檢查 gripper 是否進入桌子的碰撞範圍（長方體區域）
    """
    x, y, z = gripper_pose.position.x, gripper_pose.position.y, gripper_pose.position.z
    in_x = TABLE_BOUNDS["x"][0] <= x <= TABLE_BOUNDS["x"][1]
    in_y = TABLE_BOUNDS["y"][0] <= y <= TABLE_BOUNDS["y"][1]
    in_z = TABLE_BOUNDS["z"][0] <= z <= TABLE_BOUNDS["z"][1]
    return in_x and in_y and in_z