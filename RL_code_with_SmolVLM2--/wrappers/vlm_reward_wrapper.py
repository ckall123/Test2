# wrappers/vlm_reward_wrapper.py
from __future__ import annotations
import math
from typing import Tuple, Dict, Any

try:
    import gymnasium as gym
except Exception:  # gym 老版本相容
    import gym

import numpy as np

from vlm.sync_api import get_vlm_score


class VLMRewardWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env,
                 model: str = "qwen2.5vl",
                 interval: int = 10,
                 coeff: float = 0.5,
                 prompt: str | None = None):
        super().__init__(env)
        assert interval >= 1
        self.model = model
        self.interval = int(interval)
        self.coeff = float(coeff)
        self.prompt = prompt  # 若你要動態換 prompt，可擴充 sync_api 支援
        self._last_score = 0.0
        self._prev_score = 0.0
        self._t = 0

    # --- 影像取得 ---
    def _grab_image(self) -> np.ndarray:
        # 優先使用 env.latest_image，其次從 ROS helper 取圖
        img = getattr(self.env, "latest_image", None)
        if img is None and hasattr(self.env, "ros"):
            img = getattr(self.env.ros, "_last_image", None)
            if img is None and hasattr(self.env.ros, "get_image"):
                try:
                    img = self.env.ros.get_image(fill_if_none=True)
                except Exception:
                    img = None
        if img is None:
            # 退而求其次，從 obs 讀
            try:
                obs = self.env._last_obs  # 自訂快取；若沒有可在 reset/step 中設
                img = obs.get("image") if isinstance(obs, dict) else None
            except Exception:
                img = None
        if img is None:
            # 造一張黑圖避免炸掉
            return np.zeros((224, 224, 3), dtype=np.uint8)
        return img

    # --- Gym 介面 ---
    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        obs = out[0] if isinstance(out, tuple) else out
        try:
            self.env._last_obs = obs
        except Exception:
            pass
        img = self._grab_image()
        s = float(get_vlm_score(img, model=self.model))
        self._prev_score = s
        self._last_score = s
        self._t = 0
        return out

    def step(self, action):
        out = self.env.step(action)
        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
        else:  # gym<=0.25
            obs, reward, done, info = out
            terminated, truncated = bool(done), False
        try:
            self.env._last_obs = obs
        except Exception:
            pass

        self._t += 1
        vlm_delta = 0.0
        vlm_score = self._last_score
        if self._t % self.interval == 0:
            img = self._grab_image()
            vlm_score = float(get_vlm_score(img, model=self.model))
            vlm_delta = vlm_score - self._last_score
            self._prev_score = self._last_score
            self._last_score = vlm_score

        reward = float(reward) + self.coeff * float(vlm_delta)
        info = dict(info or {})
        info.update({"vlm_score": vlm_score, "vlm_delta": vlm_delta})

        return obs, reward, terminated, truncated, info