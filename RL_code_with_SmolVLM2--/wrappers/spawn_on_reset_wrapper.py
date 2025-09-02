"""
SpawnOnResetWrapper — v2
------------------------
用途：在每次 `env.reset()` 時，控制往 Gazebo spawn 哪些物件。
若未提供 `spawner_cfg`，會自動使用 **中央設定**（`config/table_area.DEFAULT_TABLE_AREA`）建立 Spawner。
支援兩種模式：
1) "use_specs"：每次都用你提供的 `specs` 逐一 spawn（位置/朝向仍可隨機）。
2) "sample_catalog"：從你提供的 `catalog` 隨機抽樣 N 個（N 介於 [min_n, max_n]），每次 reset 都可不同。

包裝流程：
- reset() 前先刪掉上一回合 spawn 的物件（可關閉）。
- 依模式決定本回合要 spawn 的清單，呼叫 Spawner.spawn_batch(randomize=True)。
- 把 `spawned_models`（name/xyzrpy/radius）與 `table_rect/table_z` 放進 `info` 回傳。

備註：
- 本 wrapper 與 `objects/spawner.py` 分離；請先確認 Spawner 服務 ready。
- 需要 rclpy 已初始化（與你的 env 相同假設）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
import gymnasium as gym

from objects.spawner import Spawner, SpawnerConfig, TableArea, ModelSpec, SpawnedModel

# 嘗試載入中央桌面範圍設定（可選）。若不存在則用內建 fallback。
try:
    from config.table_area import DEFAULT_TABLE_AREA  # type: ignore
except Exception:  # pragma: no cover
    DEFAULT_TABLE_AREA = TableArea(xmin=-0.30, xmax=+0.30, ymin=+0.20, ymax=+0.80, z=0.76)


@dataclass
class SpawnOnResetConfig:
    # 控制模式
    mode: str = "use_specs"            # "use_specs" 或 "sample_catalog"

    # 若 mode==use_specs
    specs: List[ModelSpec] = field(default_factory=list)

    # 若 mode==sample_catalog
    catalog: List[ModelSpec] = field(default_factory=list)
    min_n: int = 1
    max_n: int = 3
    catalog_probs: Optional[List[float]] = None  # 機率抽樣（長度需等於 catalog），None 則均勻

    # 通用
    randomize_pose: bool = True
    delete_on_reset: bool = True
    seed: Optional[int] = None


class SpawnOnResetWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, spawner_cfg: Optional[SpawnerConfig] = None, cfg: Optional[SpawnOnResetConfig] = None):
        super().__init__(env)
        self.cfg = cfg or SpawnOnResetConfig()
        # 若未提供 spawner_cfg，使用中央 DEFAULT_TABLE_AREA 建立預設 SpawnerConfig
        if spawner_cfg is None:
            spawner_cfg = SpawnerConfig(table_area=DEFAULT_TABLE_AREA)
        self.spawner_cfg = spawner_cfg
        self.spawner = Spawner(spawner_cfg)
        self.spawner.wait_until_ready()

        self._rng = np.random.default_rng(self.cfg.seed)
        self._spawned_names: List[str] = []

    # --------------------------------------------------------------
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info = info or {}

        # 清空上一回合
        if self.cfg.delete_on_reset and self._spawned_names:
            try:
                self.spawner.delete_many(self._spawned_names)
            except Exception:
                pass
            self._spawned_names = []

        # 決定這回合的 specs
        specs: List[ModelSpec]
        if self.cfg.mode == "use_specs":
            specs = list(self.cfg.specs)
        elif self.cfg.mode == "sample_catalog":
            N = int(self._rng.integers(low=self.cfg.min_n, high=self.cfg.max_n + 1))
            if len(self.cfg.catalog) == 0:
                specs = []
            else:
                idx = self._choice_without_replace(len(self.cfg.catalog), N, self.cfg.catalog_probs)
                specs = [self.cfg.catalog[i] for i in idx]
        else:
            specs = list(self.cfg.specs)

        # 進行 spawn
        seed_now = int(self._rng.integers(0, 2**31 - 1))
        spawned: List[SpawnedModel] = self.spawner.spawn_batch(specs, randomize=self.cfg.randomize_pose, seed=seed_now)
        self._spawned_names = [s.name for s in spawned]

        # info 輸出
        ar: TableArea = self.spawner_cfg.table_area
        info.update({
            "spawned_models": [{"name": s.name, "xyzrpy": s.xyzrpy, "radius": s.radius} for s in spawned],
            "table_rect": (float(ar.xmin), float(ar.xmax), float(ar.ymin), float(ar.ymax)),
            "table_z": float(ar.z),
            "spawn_seed": seed_now,
            "spawn_mode": self.cfg.mode,
        })
        return obs, info

    # --------------------------------------------------------------
    def _choice_without_replace(self, n: int, k: int, probs: Optional[List[float]] = None):
        k = max(0, min(k, n))
        if k == 0:
            return []
        if probs is None:
            return list(self._rng.choice(n, size=k, replace=False))
        p = np.asarray(probs, dtype=np.float64)
        if p.shape[0] != n:
            raise ValueError("catalog_probs length must match catalog length")
        p = p / p.sum()
        # 以機率抽樣再去重，直到湊滿 k 或達到嘗試上限
        out = set()
        tries = 0
        while len(out) < k and tries < 10 * k:
            pick = int(self._rng.choice(n, p=p))
            out.add(pick)
            tries += 1
        if len(out) < k:
            # 退而求其次：補上缺少的（均勻）
            remain = [i for i in range(n) if i not in out]
            need = k - len(out)
            if remain:
                out.update(self._rng.choice(len(remain), size=min(need, len(remain)), replace=False))
        return list(out)[:k]


__all__ = ["SpawnOnResetWrapper", "SpawnOnResetConfig"]
