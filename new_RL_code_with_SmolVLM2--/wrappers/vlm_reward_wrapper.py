from __future__ import annotations
from typing import Tuple, Dict, Any

try:
    import gymnasium as gym
except Exception:
    import gym

import numpy as np
import cv2
from vlm.sync_api import get_vlm_score


class VLMRewardWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, interval: int = 10, coeff: float = 0.5,
                 model: str = "qwen2-vl", prompt: str | None = None):
        super().__init__(env)
        self.interval = max(1, int(interval))
        self.coeff = float(coeff)
        self.model = str(model)
        self.prompt = prompt if (prompt is not None and len(prompt) > 0) else getattr(env, "vlm_prompt", None)
        self._t = 0
        self._last_score = 0.0

    def _grab_image_rgb(self):
        # 一律用 ROS 原始 224×224 RGB（避免拿到縮圖）
        ros = getattr(self.env, "ros", None)
        if ros is not None and hasattr(ros, "get_image"):
            img = ros.get_image(fill_if_none=True)
        else:
            img = getattr(self.env, "latest_image", None)
        if img is None:
            try:
                img = self.env._last_obs.get("image", None)  # type: ignore
            except Exception:
                img = None
        if img is None:
            img = np.zeros((224, 224, 3), dtype=np.uint8)
        return img

    def _rgb_to_bgr(self, img_rgb: np.ndarray) -> np.ndarray:
        try:
            return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        except Exception:
            return img_rgb[..., ::-1]

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        obs = out[0] if isinstance(out, tuple) else out
        try:
            self.env._last_obs = obs
        except Exception:
            pass
        rgb = self._grab_image_rgb()
        bgr = self._rgb_to_bgr(rgb)
        self._last_score = float(get_vlm_score(bgr, model=self.model, prompt=self.prompt))
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
            rgb = self._grab_image_rgb()
            bgr = self._rgb_to_bgr(rgb)
            vlm_score = float(get_vlm_score(bgr, model=self.model, prompt=self.prompt))
            vlm_delta = vlm_score - self._last_score
            self._last_score = vlm_score

        shaped = float(reward) + self.coeff * float(vlm_delta)
        info = dict(info or {})
        info.update({"vlm_score": vlm_score, "vlm_delta": vlm_delta})
        return obs, shaped, terminated, truncated, info