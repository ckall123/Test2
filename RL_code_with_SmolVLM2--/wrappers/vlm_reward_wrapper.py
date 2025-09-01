# =========================
# FILE: wrappers/vlm_reward_wrapper.py
# =========================
from __future__ import annotations
from typing import Tuple, Dict, Any

try:
    import gymnasium as gym
except Exception:
    import gym

import numpy as np
from vlm.sync_api import get_vlm_score


class VLMRewardWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, interval: int = 10, coeff: float = 0.5,
                 model: str = "qwen2.5vl", prompt: str = None):
        super().__init__(env)
        self.interval = max(1, int(interval))
        self.coeff = float(coeff)
        self.model = str(model)
        self.prompt = prompt if (prompt is not None and len(prompt) > 0) else getattr(env, "vlm_prompt", None)
        self._t = 0
        self._last_score = 0.0

    def _grab_image(self):
        img = getattr(self.env, "latest_image", None)
        if img is None:
            ros = getattr(self.env, "ros", None)
            if ros is not None and hasattr(ros, "get_image"):
                img = ros.get_image(fill_if_none=True)
        if img is None:
            try:
                img = self.env._last_obs.get("image", None)  # type: ignore
            except Exception:
                img = None
        if img is None:
            return np.zeros((224, 224, 3), dtype=np.uint8)
        return img

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        obs = out[0] if isinstance(out, tuple) else out
        try:
            self.env._last_obs = obs
        except Exception:
            pass
        img = self._grab_image()
        self._last_score = float(get_vlm_score(img, model=self.model, prompt=self.prompt))
        self._t = 0
        return out

    def step(self, action):
        out = self.env.step(action)
        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
        else:
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
            vlm_score = float(get_vlm_score(img, model=self.model, prompt=self.prompt))
            vlm_delta = vlm_score - self._last_score
            self._last_score = vlm_score

        shaped = float(reward) + self.coeff * float(vlm_delta)
        info = dict(info or {})
        info.update({"vlm_score": vlm_score, "vlm_delta": vlm_delta})
        return obs, shaped, terminated, truncated, info
