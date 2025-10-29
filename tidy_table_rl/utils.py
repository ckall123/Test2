# utils.py
# 通用輔助工具（與 ROS/MoveIt 無關）
# - 環境設定 EnvConfig
# - 影像處理 preprocess_image
# - 動作歷史 ActionHistory
# - 動作懲罰 compute_action_cost
# - VLM 控制加分 VLMThrottle
# - 距離分段獎勵 distance_sparse_bonus

from dataclasses import dataclass
from collections import deque
from typing import Deque, Optional, Tuple, Sequence

import numpy as np
import cv2


# =====================================
# 🧩 環境設定 Config
# =====================================
@dataclass
class EnvConfig:
    # 基本參數
    image_size: Tuple[int, int] = (96, 96)
    max_steps: int = 400
    action_scale: float = 0.08

    # 幾何能量權重
    w_pca: float = 1.0
    w_spacing: float = 1.0
    w_yaw: float = 0.5
    w_overlap: float = 2.0
    w_out: float = 1.5
    overlap_lambda: float = 1.1
    edge_margin: float = 0.03
    E_success: float = 0.10

    # 目標距離成功判定（Done 條件可用）
    success_dist: float = 0.01

    # 接觸一次性加分
    contact_bonus: float = 0.2

    # 懲罰設計
    prox_weight: float = 0.5
    action_hist_len: int = 10
    action_cost_coef: float = 0.01
    collision_penalty: float = 1.0

    # VLM 加分控制
    vlm_interval: int = 8
    vlm_deltaE_thresh: float = 0.02
    vlm_clip: float = 0.5

    # 距離里程碑（分段獎勵）
    distance_bins: Tuple[float, ...] = (0.05, 0.04, 0.03, 0.02, 0.01)
    distance_vals: Tuple[float, ...] = (0.05, 0.15, 0.30, 0.60, 1.00)


# =====================================
# 🖼️ 影像處理
# =====================================
def preprocess_image(image_rgb: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """
    將相機影像縮放為指定尺寸。
    :param image_rgb: HxWx3 RGB uint8
    :param size: (W, H)
    :return: 縮放後影像 HxWx3 uint8
    """
    if image_rgb is None:
        raise ValueError("image_rgb is None")
    if image_rgb.dtype != np.uint8:
        image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("image_rgb 必須是 HxWx3")

    w, h = size
    return cv2.resize(image_rgb, (w, h), interpolation=cv2.INTER_AREA)


# =====================================
# 🔁 動作歷史記錄
# =====================================
class ActionHistory:
    """
    儲存最近 N 次動作，提供展平向量輸出。
    """
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
            seq = np.stack(list(self._dq)[-k:], axis=0)
            out[-k:] = seq
        return out.reshape(-1)


# =====================================
# ⚠️ 動作懲罰（正則化）
# =====================================
def compute_action_cost(action: np.ndarray | Sequence[float], coef: float) -> float:
    """
    L2 動作懲罰，防止震盪。
    """
    a = np.asarray(action, dtype=np.float32)
    return float(coef * np.linalg.norm(a))


# =====================================
# 🌟 VLM 節流控制（多步加分）
# =====================================
class VLMThrottle:
    """
    控制是否觸發 VLM 打分（減少頻率）
    - reset(init_score) 重設初始分數
    - maybe_bonus(...) 滿足條件才呼叫 VLM 評分
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


# =====================================
# 🧠 里程碑工具（距離/接觸）
# =====================================

def validate_milestones(bins: Sequence[float], vals: Sequence[float]):
    """將距離門檻與獎勵排序為『距離由小到大、獎勵不遞減』並檢查有效性。"""
    b = np.asarray(bins, dtype=float)
    v = np.asarray(vals, dtype=float)
    if b.size != v.size:
        raise ValueError("distance_bins 與 distance_vals 長度必須相同")
    order = np.argsort(b)  # 距離小→大
    b, v = b[order], v[order]
    if not np.all(np.diff(v) >= 0):
        raise ValueError("distance_vals 必須對應為非遞減（越近獎勵不會變小）")
    return b, v


class DistanceMilestone:
    """
    分段距離獎勵（每段只給一次）。
    - reset()：清除已領取的段位
    - update(d)：若首次達成更嚴格的距離段位，回傳該段位獎勵，否則 0
    注意：bins/vals 會自動按距離由小到大排序（0.01, 0.02, ...）
    """
    def __init__(self, bins: Sequence[float], vals: Sequence[float]):
        self.bins, self.vals = validate_milestones(bins, vals)
        self._claimed = np.zeros_like(self.bins, dtype=bool)

    def reset(self):
        self._claimed[:] = False

    def update(self, d: float) -> float:
        # 從最嚴格（最小距離）往外檢查，命中第一個未領取門檻就給該段位獎勵
        for i in range(len(self.bins)):
            if d < self.bins[i] and not self._claimed[i]:
                self._claimed[i] = True
                return float(self.vals[i])
        return 0.0


class ContactOnceBonus:
    """
    接觸一次性加分：同一物件在單一 episode 只加分一次。
    可選擇在加分同時觸發 side-effect（例如 attach 物件）。
    """
    def __init__(self, bonus: float, on_first_contact=None):
        self.bonus = float(bonus)
        self.on_first_contact = on_first_contact  # 可傳入 callable(name: str)
        self._seen: set[str] = set()

    def reset(self):
        self._seen.clear()

    def award_if_first(self, name: str) -> float:
        if name in self._seen:
            return 0.0
        self._seen.add(name)
        if callable(self.on_first_contact):
            try:
                self.on_first_contact(name)
            except Exception:
                pass
        return self.bonus


# =====================================
# 🪜 距離分段獎勵
# =====================================
def distance_milestone_reward(d: float, bins: Sequence[float], vals: Sequence[float]) -> float:
    """回傳『目前距離所能達到的最高段位獎勵』；若未命中任何段位則回傳 0。"""
    b, v = validate_milestones(bins, vals)
    # 找到最嚴格（最小門檻）但仍滿足 d < b[i] 的段位；以高段位為優先
    for i in range(len(b)):
        if d < b[i]:
            return float(v[i])
    return 0.0


def distance_sparse_bonus(d: float, bins: Sequence[float], vals: Sequence[float]) -> float:
    """
    [Deprecated]：請改用 `distance_milestone_reward()` 或狀態化的 `DistanceMilestone`。
    為向後相容，這裡回傳與 `distance_milestone_reward` 相同邏輯之結果。
    """
    return distance_milestone_reward(d, bins, vals)


# =====================================
# 🧰 薄介面：讓外部程式更好用（Functional Facade）
# =====================================
from typing import Callable, NamedTuple

class RewardHelpers(NamedTuple):
    """回傳給使用者的一組『一步到位』的函式與狀態容器。"""
    reset: Callable[[], None]
    distance_bonus: Callable[[float], float]
    contact_bonus: Callable[[str], float]
    vlm_bonus: Callable[[float, np.ndarray, object], float]
    action_cost: Callable[[np.ndarray], float]
    hist: ActionHistory
    hist_vec: Callable[[], np.ndarray]


def new_episode_helpers(cfg: EnvConfig, attach: Optional[Callable[[str], None]] = None) -> RewardHelpers:
    """
    建立即開即用的輔助器：
    - 距離分段獎勵（每段只發一次）
    - 夾爪雙指接觸一次性獎勵（同物件單回合只發一次，可選擇同步 attach）
    - VLM 節流加分（需傳入 ΔE 與當前畫面）
    - 動作成本（L2）
    - 動作歷史（可直接取 flattened 向量）

    用法：
        helpers = new_episode_helpers(cfg, attach_fn)
        helpers.reset()
        r += helpers.distance_bonus(d)
        r += helpers.contact_bonus(target_name)
        r += helpers.vlm_bonus(delta_E, img_rgb, vlm)
        r -= helpers.action_cost(action)
        obs_hist = helpers.hist_vec()
    """
    _dist = DistanceMilestone(cfg.distance_bins, cfg.distance_vals)
    _contact = ContactOnceBonus(cfg.contact_bonus, on_first_contact=attach)
    _vlm = VLMThrottle(cfg.vlm_interval, cfg.vlm_deltaE_thresh, cfg.vlm_clip)
    _hist = ActionHistory(cfg.action_hist_len)

    def _reset():
        _dist.reset()
        _contact.reset()
        _vlm.reset(0.0)
        _hist.clear()

    def _distance_bonus(d: float) -> float:
        return _dist.update(float(d))

    def _contact_bonus(name: str) -> float:
        return _contact.award_if_first(str(name))

    def _vlm_bonus(delta_E: float, image_rgb: np.ndarray, vlm) -> float:
        return _vlm.maybe_bonus(float(delta_E), image_rgb, vlm)

    def _action_cost(a: np.ndarray) -> float:
        return compute_action_cost(a, cfg.action_cost_coef)

    return RewardHelpers(
        reset=_reset,
        distance_bonus=_distance_bonus,
        contact_bonus=_contact_bonus,
        vlm_bonus=_vlm_bonus,
        action_cost=_action_cost,
        hist=_hist,
        hist_vec=_hist.vector,
    )


# --- 無狀態版（你只想一行算當下可拿到的距離獎勵） ---

def distance_bonus_now(d: float, bins: Sequence[float], vals: Sequence[float]) -> float:
    return distance_milestone_reward(d, bins, vals)


def contact_bonus_once(seen: set, name: str, bonus: float, attach: Optional[Callable[[str], None]] = None) -> float:
    """無類別版本：用一個 set 記錄本回合碰過的物件。"""
    if name in seen:
        return 0.0
    seen.add(name)
    if callable(attach):
        try:
            attach(name)
        except Exception:
            pass
    return float(bonus)
