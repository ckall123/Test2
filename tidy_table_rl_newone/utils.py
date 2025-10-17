#!/usr/bin/env python3
"""
utils.py
通用輔助工具（與 ROS/MoveIt 無關）：
- EnvConfig：集中管理環境/獎勵參數
- 影像前處理：preprocess_image()
- 動作歷史：ActionHistory（支援預先指定 action_dim）
- 動作成本：compute_action_cost()
- VLM 節流：VLMThrottle（步距、ΔE 門檻、分數差夾限）
- 距離里程碑：distance_sparse_bonus()
"""

from dataclasses import dataclass
from collections import deque
from typing import Deque, Optional, Tuple, Sequence

import numpy as np
import cv2


# =========================
# Config
# =========================
@dataclass
class EnvConfig:
    # 基本流程
    image_size: Tuple[int, int] = (96, 96)  # (W, H)
    max_steps: int = 400
    action_scale: float = 0.08

    # 幾何能量 E 權重/參數
    w_pca: float = 1.0
    w_spacing: float = 1.0
    w_yaw: float = 0.5
    w_overlap: float = 2.0
    w_out: float = 1.5
    overlap_lambda: float = 1.1
    edge_margin: float = 0.03
    E_success: float = 0.10

    # 形狀成形與懲罰
    prox_weight: float = 0.5
    action_hist_len: int = 10
    action_cost_coef: float = 0.01
    collision_penalty: float = 1.0

    # VLM 受控加分
    vlm_interval: int = 8
    vlm_deltaE_thresh: float = 0.02
    vlm_clip: float = 0.5

    # 距離里程碑（公尺）：對應獎勵（同索引）
    distance_bins: Tuple[float, ...] = (0.05, 0.04, 0.03, 0.02, 0.01)
    distance_vals: Tuple[float, ...] = (0.05, 0.15, 0.30, 0.60, 1.00)


# =========================
# Image
# =========================
def preprocess_image(image_rgb: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """將相機 RGB 影像縮放到 (W,H)，回傳 HxWx3 uint8（Env 內再做通道轉置）。"""
    if image_rgb is None:
        raise ValueError("image_rgb is None")
    img = image_rgb
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("image_rgb 必須是 HxWx3")
    w, h = size
    out = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    return out


# =========================
# Action history
# =========================
class ActionHistory:
    """維護最近 N 次動作；vector() 輸出固定長度展平向量（N * action_dim）。"""
    def __init__(self, max_len: int, action_dim: Optional[int] = None):
        self.max_len = int(max_len)
        self._dq: Deque[np.ndarray] = deque(maxlen=self.max_len)
        self._action_dim: Optional[int] = action_dim

    def clear(self) -> None:
        self._dq.clear()

    def add(self, action: np.ndarray | Sequence[float]) -> None:
        a = np.asarray(action, dtype=np.float32)
        if self._action_dim is None:
            self._action_dim = int(a.size)
        elif a.size != self._action_dim:
            raise ValueError(f"action size {a.size} != expected {self._action_dim}")
        self._dq.append(a)

    def vector(self) -> np.ndarray:
        if self._action_dim is None:
            return np.zeros(0, dtype=np.float32)
        out = np.zeros((self.max_len, self._action_dim), dtype=np.float32)
        k = min(len(self._dq), self.max_len)
        if k > 0:
            seq = np.stack(list(self._dq)[-k:], axis=0)  # (k, action_dim)
            out[-k:] = seq
        return out.reshape(-1)


# =========================
# Regularization
# =========================
def compute_action_cost(action: np.ndarray | Sequence[float], coef: float) -> float:
    """L2 動作成本（小幅正則，避免震盪）。"""
    a = np.asarray(action, dtype=np.float32)
    return float(coef * np.linalg.norm(a))


# =========================
# VLM throttle & distance milestones
# =========================
class VLMThrottle:
    """
    管理 VLM 受控加分：reset(init_score) → maybe_bonus(delta_E, image_rgb, vlm)
    - 僅在 ΔE 超過門檻，且步距達標時，請 VLM 打分並回傳分數差（夾限）。
    """
    def __init__(self, interval: int, deltaE_thresh: float, clip: float):
        self.interval = int(interval)
        self.deltaE_thresh = float(deltaE_thresh)
        self.clip = float(clip)
        self.steps = 0
        self.last_score = 0.0

    def reset(self, init_score: float = 0.0) -> None:
        self.steps = 0
        self.last_score = float(init_score)

    def maybe_bonus(self, delta_E: float, image_rgb: np.ndarray, vlm) -> float:
        self.steps += 1
        if not (delta_E > self.deltaE_thresh and self.steps >= self.interval):
            return 0.0
        s = float(vlm.score_image(image_rgb, instruction="align objects in a row"))
        dv = float(np.clip(s - self.last_score, -self.clip, self.clip))
        self.last_score = s
        self.steps = 0
        return dv


def distance_sparse_bonus(d: float, bins: Sequence[float], vals: Sequence[float]) -> float:
    """距離里程碑加分：命中第一個閾值即回傳對應獎勵；否則 0。"""
    for th, v in zip(bins, vals):
        if d < th:
            return float(v)
    return 0.0


# =========================
# Self-test
# =========================
if __name__ == '__main__':
    # Image
    dummy = np.random.randint(0, 256, size=(240, 320, 3), dtype=np.uint8)
    out = preprocess_image(dummy, (96, 96))
    print("[image]", dummy.shape, "->", out.shape)

    # ActionHistory (7 維示例)
    hist = ActionHistory(max_len=3, action_dim=7)
    print("[hist] empty:", hist.vector().shape)
    hist.add(np.arange(7, dtype=np.float32))
    hist.add(np.ones(7, dtype=np.float32))
    print("[hist] vec:", hist.vector().shape)

    # VLMThrottle（以假 VLM 測流程）
    class _DummyVLM:
        def __init__(self): self.v = 0.4
        def score_image(self, *_a, **_k): self.v += 0.1; return min(self.v, 1.0)
    vt = VLMThrottle(interval=2, deltaE_thresh=0.01, clip=0.5)
    vt.reset(0.2)
    for i in range(5):
        bonus = vt.maybe_bonus(delta_E=0.02, image_rgb=out, vlm=_DummyVLM())
        print(f"[vlm] step {i} bonus:", bonus)

    # Distance milestones
    b = distance_sparse_bonus(0.028, (0.05, 0.04, 0.03, 0.02, 0.01), (0.05, 0.15, 0.30, 0.60, 1.00))
    print("[dist bonus]", b)
