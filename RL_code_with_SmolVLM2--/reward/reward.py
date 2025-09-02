"""
Reward utilities (minimal)
- Safety: compute counts from contact pairs and a smooth safety score.
- Layout: light placeholders so __init__ exports won't break (you can ignore for now).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import math

Pair = Tuple[str, str, str, str]


# ---------------- Safety ----------------
@dataclass
class SafetyWeights:
    w_self: float = 1.0
    w_table: float = 0.5
    w_arm_nongrip: float = 0.25


def _is_table(model: str, table_models: Sequence[str]) -> bool:
    m = model.lower()
    return any(m == t.lower() for t in table_models)


def safety_score_from_contacts(
    pairs: Iterable[Pair],
    *,
    robot_model: str,
    gripper_links: Sequence[str],
    table_models: Sequence[str] = ("table", "ground", "ground_plane", "world"),
    attached_model: Optional[str] = None,
    weights: SafetyWeights | None = None,
) -> Tuple[float, Dict[str, int]]:
    """Return (safety_score in [0,1], counts) from contact name pairs.

    counts keys:
      - self_collision: robot vs robot
      - robot_table: robot vs table/world/ground
      - arm_obj_nongrip: robot (non-gripper links) vs non-table, non-attached object
    """
    weights = weights or SafetyWeights()
    robot_model_l = robot_model.lower()
    grip_set = {g.lower() for g in gripper_links}
    attached_l = (attached_model or "").lower()

    c_self = 0
    c_table = 0
    c_arm = 0

    def norm_pair(p: Pair) -> Pair:
        m1, l1, m2, l2 = p
        return m1 or "", l1 or "", m2 or "", l2 or ""

    for p in pairs:
        m1, l1, m2, l2 = norm_pair(p)
        ml1, ll1, ml2, ll2 = m1.lower(), l1.lower(), m2.lower(), l2.lower()

        is_r1 = (ml1 == robot_model_l)
        is_r2 = (ml2 == robot_model_l)
        if is_r1 and is_r2:
            c_self += 1
            continue

        # robot vs table/world
        if (is_r1 and _is_table(ml2, table_models)) or (is_r2 and _is_table(ml1, table_models)):
            c_table += 1
            continue

        # robot (non-gripper link) vs object that's not attached and not table
        if is_r1 and not is_r2:
            other_m, link = ml2, ll1
        elif is_r2 and not is_r1:
            other_m, link = ml1, ll2
        else:
            continue  # object-object contacts ignored

        if other_m == attached_l or _is_table(other_m, table_models):
            continue
        if link not in grip_set:
            c_arm += 1

    # Smooth score in [0,1]
    cost = weights.w_self * c_self + weights.w_table * c_table + weights.w_arm_nongrip * c_arm
    safety = math.exp(-float(cost))
    counts = {"self_collision": c_self, "robot_table": c_table, "arm_obj_nongrip": c_arm}
    return float(safety), counts


# ---------------- Layout (placeholders for now) ----------------
@dataclass
class LayoutWeights:
    dist_align: float = 1.0
    cluster_pen: float = 0.0


def layout_score_positions(
    positions_xy: Sequence[Tuple[float, float]],
    radii: Sequence[float],
    table_rect: Tuple[float, float, float, float],
    weights: LayoutWeights | None = None,
) -> float:
    """Very light placeholder: returns a small penalty if any two circles overlap.
    1.0 means good (no overlaps), lower means worse.
    """
    weights = weights or LayoutWeights()
    n = len(positions_xy)
    overlap = 0
    for i in range(n):
        xi, yi = positions_xy[i]
        ri = radii[i]
        for j in range(i + 1, n):
            xj, yj = positions_xy[j]
            rj = radii[j]
            d2 = (xi - xj) ** 2 + (yi - yj) ** 2
            if d2 < (ri + rj) ** 2:
                overlap += 1
    return float(math.exp(-weights.cluster_pen * overlap))


def combine_geom_with_vlm_delta(geom_score: float, vlm_delta: float, alpha: float = 0.5) -> float:
    return float(alpha * geom_score + (1.0 - alpha) * vlm_delta)


def combine_geom_safety_vlm(
    geom_score: float,
    safety_score: float,
    vlm_delta: float,
    w_geom: float = 0.4,
    w_safety: float = 0.4,
    w_vlm: float = 0.2,
) -> float:
    return float(w_geom * geom_score + w_safety * safety_score + w_vlm * vlm_delta)
