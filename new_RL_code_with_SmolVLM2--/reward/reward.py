"""reward/reward.py

獎勵模組（乾淨分離）：
- 幾何獎勵（靠近物體 + 工作區 Z 高度判斷掉落）
- 版面整齊度（Y 對齊 + X 間距穩定）
- 與 VLM 分數融合的最終獎勵
- 觸碰桌面偵測與懲罰（含觸碰容忍次數機制）
- 打包好的 `RewardComponents` 與 `compute_rewards(...)` 一次算好所有欄位

保留向後相容的 API：geom_reward / layout_score / final_reward / touched_table / penalty_touch_table
並新增 RewardComponents / compute_rewards 供更高層直接使用。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable, Dict, Any, Optional
import numpy as np

from objects.xarm_positions import get_gripper_span


# -------------------------------
# 資料結構：獎勵各組件
# -------------------------------
@dataclass
class RewardComponents:
    geom: float = 0.0
    layout: float = 0.0
    vlm: float = 0.0
    touched: bool = False
    final: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# -------------------------------
# 幾何獎勵（靠近物體 + 檢查 Z 掉落）
# -------------------------------

def geom_reward(
    ee: np.ndarray,
    objects: Iterable[np.ndarray],
    workspace: Dict[str, tuple],
    *,
    distance_scale: float = 0.5,
    out_of_workspace_penalty: float = 0.5,
) -> float:
    obj_list = list(objects)
    if len(obj_list) == 0:
        return 0.0

    P = np.stack(obj_list, axis=0)
    dists = np.linalg.norm(P - ee, axis=1)
    r_geom = max(0.0, 1.0 - float(dists.min()) / max(1e-6, distance_scale))

    # 僅用 Z 掉落懲罰（X/Y 不判斷）
    for p in P:
        if p[2] < 0.0:
            r_geom -= out_of_workspace_penalty

    return float(r_geom)


# -------------------------------
# 版面整齊度（橫向對齊 + 間距一致）
# -------------------------------

def layout_score(
    objects: Iterable[np.ndarray],
    *,
    y_std_scale: float = 5.0,
    gap_var_scale: float = 10.0,
    y_weight: float = 0.5,
    gap_weight: float = 0.5,
) -> float:
    obj_list = list(objects)
    if len(obj_list) < 2:
        return 0.0

    P = np.stack(obj_list, axis=0)
    y_std = float(np.std(P[:, 1]))
    xs = np.sort(P[:, 0])
    gaps = np.diff(xs)
    gap_var = float(np.var(gaps)) if len(gaps) > 0 else 0.0

    y_score = float(np.exp(-y_std_scale * y_std))
    gap_score = float(np.exp(-gap_var_scale * gap_var))
    return y_weight * y_score + gap_weight * gap_score


# -------------------------------
# 視覺語言分數融合（+ layout + geom）
# -------------------------------

def final_reward(
    r_geom: float,
    r_layout: float,
    vlm_score: float,
    *,
    w_geom: float = 0.6,
    w_layout: float = 0.4,
    w_mix: float = 0.7,
    w_vlm: float = 0.3,
) -> float:
    base = w_geom * float(r_geom) + w_layout * float(r_layout)
    return float(w_mix * base + w_vlm * float(vlm_score))


# -------------------------------
# 碰桌子偵測（含容忍次數）
# -------------------------------

class TouchCounter:
    def __init__(self, max_touch: int = 5):
        self.count = 0
        self.max_touch = max_touch

    def update(self, touched: bool):
        if touched:
            self.count += 1
        else:
            self.count = 0

    def triggered(self) -> bool:
        return self.count >= self.max_touch


def touched_table(tf_buffer, z_thresh: float = 0.03, timeout_sec: float = 0.05) -> bool:
    try:
        span = get_gripper_span(tf_buffer, reference_frame="world", timeout_sec=timeout_sec)
        return bool(span.get("z_center", 1.0) < z_thresh)
    except Exception:
        return False


def penalty_touch_table(tf_buffer, z_thresh: float = 0.03) -> float:
    return -1.0 if touched_table(tf_buffer, z_thresh=z_thresh) else 0.0


# -------------------------------
# 統一 Done 條件：Z 掉落 or 桌碰撞
# -------------------------------

def check_done(
    objects: Iterable[np.ndarray],
    ee: np.ndarray,
    touch_counter: TouchCounter,
    *,
    table_z: float = 0.0,
    drop_tol: float = 0.05,
) -> tuple[bool, dict]:
    info = {}
    done = False

    for obj in objects:
        if obj[2] < (table_z - drop_tol):
            info['reason'] = 'object_fell'
            return True, info

    if touch_counter.triggered():
        info['reason'] = 'table_collision'
        return True, info

    return False, info


# -------------------------------
# 一次把所有獎勵算好：compute_rewards
# -------------------------------

def compute_rewards(
    ee: np.ndarray,
    objects: Iterable[np.ndarray],
    tf_buffer,
    vlm_score: float,
    workspace: Dict[str, tuple],
    *,
    weights: Optional[Dict[str, float]] = None,
    distance_scale: float = 0.5,
    out_of_workspace_penalty: float = 0.5,
    y_std_scale: float = 5.0,
    gap_var_scale: float = 10.0,
    y_weight: float = 0.5,
    gap_weight: float = 0.5,
    touch_penalty: float = -1.0,
) -> RewardComponents:
    w = {
        'w_geom': 0.6,
        'w_layout': 0.4,
        'w_mix': 0.7,
        'w_vlm': 0.3,
    }
    if weights:
        w.update({k: float(v) for k, v in weights.items() if k in w})

    r = RewardComponents()

    r.geom = geom_reward(
        ee,
        objects,
        workspace,
        distance_scale=distance_scale,
        out_of_workspace_penalty=out_of_workspace_penalty,
    )
    r.layout = layout_score(
        objects,
        y_std_scale=y_std_scale,
        gap_var_scale=gap_var_scale,
        y_weight=y_weight,
        gap_weight=gap_weight,
    )
    r.vlm = float(vlm_score)

    r.touched = touched_table(tf_buffer)
    r.final = final_reward(
        r.geom,
        r.layout,
        r.vlm,
        w_geom=w['w_geom'],
        w_layout=w['w_layout'],
        w_mix=w['w_mix'],
        w_vlm=w['w_vlm'],
    )
    if r.touched and touch_penalty != 0.0:
        r.final += float(touch_penalty)

    return r


__all__ = [
    'RewardComponents',
    'geom_reward',
    'layout_score',
    'final_reward',
    'touched_table',
    'penalty_touch_table',
    'compute_rewards',
    'check_done',
    'TouchCounter',
]
