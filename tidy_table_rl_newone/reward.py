#!/usr/bin/env python3
"""
reward.py
計算桌面整齊度的幾何能量 E，供強化學習作為主 reward。
"""

from typing import List, Dict, Tuple, Any
import numpy as np
import json
import rclpy
from pose_tracker import PoseTracker

def compute_geometry_energy(
    objs: List[Dict[str, Any]],
    table_bounds: Tuple[Tuple[float, float], Tuple[float, float]],
    cfg: Any,
) -> float:
    if len(objs) == 0:
        return 0.0

    def _get(name, default):
        return getattr(cfg, name, cfg.get(name, default) if isinstance(cfg, dict) else default)

    w_pca = _get("w_pca", 1.0)
    w_spacing = _get("w_spacing", 1.0)
    w_yaw = _get("w_yaw", 0.5)
    w_overlap = _get("w_overlap", 2.0)
    w_out = _get("w_out", 1.5)
    overlap_lambda = _get("overlap_lambda", 1.1)
    edge_margin = _get("edge_margin", 0.03)

    (xmin, xmax), (ymin, ymax) = table_bounds
    pts_xy = np.array([o["pos"][:2] for o in objs], dtype=float)
    yaws = np.array([o["yaw"] for o in objs], dtype=float)
    radii = np.array([o["radius"] for o in objs], dtype=float)

    if len(pts_xy) >= 2:
        C = pts_xy - pts_xy.mean(axis=0, keepdims=True)
        _, _, vh = np.linalg.svd(C, full_matrices=False)
        axis = vh[0]
        c = pts_xy.mean(axis=0)
        ortho = pts_xy - ((pts_xy - c) @ axis)[:, None] * axis
        pca_term = float((ortho ** 2).sum(axis=1).mean())
        proj_1d = ((pts_xy - c) @ axis)
        gaps = np.diff(np.sort(proj_1d))
        spacing_term = float(gaps.var()) if gaps.size > 0 else 0.0
    else:
        pca_term = spacing_term = 0.0

    yaw_term = float(np.mean(1.0 - np.cos(yaws - np.mean(yaws)))) if yaws.size >= 2 else 0.0

    overlap_term = 0.0
    for i in range(len(pts_xy)):
        for j in range(i + 1, len(pts_xy)):
            dij = float(np.linalg.norm(pts_xy[i] - pts_xy[j]))
            thr = float((radii[i] + radii[j]) * overlap_lambda)
            if dij < thr:
                overlap_term += (thr - dij)

    out_pen = 0.0
    safe_xmin, safe_xmax = xmin + edge_margin, xmax - edge_margin
    safe_ymin, safe_ymax = ymin + edge_margin, ymax - edge_margin
    for p in pts_xy:
        dx = max(safe_xmin - p[0], 0.0) if p[0] < safe_xmin else max(p[0] - safe_xmax, 0.0)
        dy = max(safe_ymin - p[1], 0.0) if p[1] < safe_ymin else max(p[1] - safe_ymax, 0.0)
        out_pen += (dx + dy)

    E = (
        w_pca * pca_term
        + w_spacing * spacing_term
        + w_yaw * yaw_term
        + w_overlap * overlap_term
        + w_out * out_pen
    )
    return float(E)


if __name__ == '__main__':
    class _Cfg:
        w_pca, w_spacing, w_yaw, w_overlap, w_out = 1.0, 1.0, 0.5, 2.0, 1.5
        overlap_lambda, edge_margin = 1.1, 0.03

    rclpy.init()
    node = rclpy.create_node("energy_debug")
    tracker = PoseTracker(node)

    with open("config.json", "r") as f:
        config = json.load(f)

    target_objects = config.get("target_objects", [])
    radius_lookup = {o["name"]: o.get("radius", 0.025) for o in config.get("objects", [])}

    for _ in range(50):
        if tracker.get_tcp_pose() is not None:
            break
        rclpy.spin_once(node, timeout_sec=0.1)

    objs = tracker.get_object_states(target_objects, radius_lookup)
    table_bounds = ((-0.30, 0.30), (-0.20, 0.20))
    E = compute_geometry_energy(objs, table_bounds, _Cfg())
    print(f"🧪 Geometry Energy E = {E:.6f}")

    node.destroy_node()
    rclpy.shutdown()