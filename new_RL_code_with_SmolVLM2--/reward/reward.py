import numpy as np
from typing import Iterable
from objects.xarm_positions import get_gripper_span

# -------------------------------
# 幾何獎勵（靠近物體 + 工作區內）
# -------------------------------
def geom_reward(ee: np.ndarray, objects: Iterable[np.ndarray], workspace: dict) -> float:
    """
    幾何部分：越靠近任一物體越高分，但物體離開工作區會扣分。
    - ee: 末端執行器位置 (3,)
    - objects: 可疊代的物體位置清單 (每個是 [x,y,z])
    - workspace: {'x':(min,max), 'y':(...), 'z':(...)}
    """
    obj_list = list(objects)
    if len(obj_list) == 0:
        return 0.0

    P = np.stack(obj_list, axis=0)
    dists = np.linalg.norm(P - ee, axis=1)
    r_geom = max(0.0, 1.0 - float(dists.min()) / 0.5)

    # 越界懲罰（每個越界扣 0.5）
    for p in P:
        x_ok = workspace["x"][0] <= p[0] <= workspace["x"][1]
        y_ok = workspace["y"][0] <= p[1] <= workspace["y"][1]
        z_ok = p[2] >= 0.0  # z 不用 upper bound
        if not (x_ok and y_ok and z_ok):
            r_geom -= 0.5

    return float(r_geom)

# -------------------------------
# 版面整齊度（橫向對齊 + 間距一致）
# -------------------------------
def layout_score(objects: Iterable[np.ndarray]) -> float:
    """
    整齊度計分：Y軸靠齊 + X軸間距平均
    - 適合用於橫向擺放的物品（如整齊排一排）
    - 適用 2 件以上物品，否則直接給 0 分
    """
    obj_list = list(objects)
    if len(obj_list) < 2:
        return 0.0

    P = np.stack(obj_list, axis=0)
    y_std = float(np.std(P[:,1]))  # Y 越一致越小
    xs = np.sort(P[:,0])
    gaps = np.diff(xs)
    gap_var = float(np.var(gaps)) if len(gaps) > 0 else 0.0

    # exp 衰減：越靠齊越接近 1.0
    return 0.5*np.exp(-5.0*y_std) + 0.5*np.exp(-10.0*gap_var)

# -------------------------------
# 視覺語言分數融合（+ layout + geom）
# -------------------------------
def final_reward(r_geom: float, r_layout: float, vlm_score: float,
                 w_geom=0.6, w_layout=0.4, w_mix=0.7, w_vlm=0.3) -> float:
    """
    組合分數：
    - 先混合幾何 + 版面：base = w_geom * geom + w_layout * layout
    - 再混進 VLM 評分：reward = w_mix * base + w_vlm * vlm
    """
    base = w_geom * float(r_geom) + w_layout * float(r_layout)
    return float(w_mix * base + w_vlm * float(vlm_score))

# -------------------------------
# 碰桌子邏輯（分成兩層用途）
# -------------------------------
def touched_table(tf_buffer, z_thresh: float = 0.03, timeout_sec: float = 0.05) -> bool:
    """
    回傳是否碰到桌子（夾爪中心 Z 值過低）
    """
    try:
        span = get_gripper_span(tf_buffer, reference_frame="world", timeout_sec=timeout_sec)
        return span["z_center"] < z_thresh
    except Exception:
        return False

def penalty_touch_table(tf_buffer, z_thresh: float = 0.03) -> float:
    """
    若碰桌子就扣分（-1.0），沒碰就 0 分。
    """
    return -1.0 if touched_table(tf_buffer, z_thresh=z_thresh) else 0.0