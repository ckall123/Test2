import numpy as np
from typing import Iterable

# 幾何距離：越靠近任何一物體越高，且越界扣分

def geom_reward(ee: np.ndarray, objects: Iterable[np.ndarray], workspace: dict) -> float:
    P = np.stack(list(objects), axis=0) if len(list(objects)) > 0 else np.zeros((0,3), np.float32)
    if P.shape[0] == 0:
        return 0.0
    dists = np.linalg.norm(P - ee, axis=1)
    r_geom = max(0.0, 1.0 - float(dists.min()) / 0.5)
    # 越界懲罰
    for p in P:
        x_ok = workspace["x"][0] <= p[0] <= workspace["x"][1]
        y_ok = workspace["y"][0] <= p[1] <= workspace["y"][1]
        z_ok = p[2] >= 0.0
        if not (x_ok and y_ok and z_ok):
            r_geom -= 0.5
    return float(r_geom)

# 版面整齊度：y 對齊 + 間距一致

def layout_score(objects: Iterable[np.ndarray]) -> float:
    P = np.stack(list(objects), axis=0)
    y_std = float(np.std(P[:,1]))
    xs = np.sort(P[:,0])
    gaps = np.diff(xs)
    gap_var = float(np.var(gaps)) if len(gaps) > 0 else 0.0
    return 0.5*np.exp(-5.0*y_std) + 0.5*np.exp(-10.0*gap_var)

# 組合最終獎勵：w = 0.7*(0.6*geom + 0.4*layout) + 0.3*vlm

def final_reward(r_geom: float, r_layout: float, vlm_score: float,
                 w_geom=0.6, w_layout=0.4, w_mix=0.7, w_vlm=0.3) -> float:
    base = w_geom*float(r_geom) + w_layout*float(r_layout)
    return float(w_mix*base + w_vlm*float(vlm_score))