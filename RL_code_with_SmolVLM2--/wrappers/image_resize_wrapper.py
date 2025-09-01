# =========================
# FILE: wrappers/image_resize_wrapper.py
# =========================
from __future__ import annotations
from typing import Tuple

try:
    import gymnasium as gym
except Exception:
    import gym

import numpy as np
import cv2


class ImageResizeWrapper(gym.ObservationWrapper):
    """
    將 obs['image'] 下採樣到固定尺寸，降低 ReplayBuffer 記憶體占用。
    預設 96x96，uint8。
    """
    def __init__(self, env: gym.Env, size: Tuple[int, int] = (96, 96)):
        super().__init__(env)
        self.size = (int(size[0]), int(size[1]))

        space = self.observation_space
        if isinstance(space, dict) or hasattr(space, "spaces"):
            # Dict 空間：維持其他鍵不變，只改 image
            spaces = dict(space.spaces)
            h, w = self.size
            spaces["image"] = gym.spaces.Box(low=0, high=255, shape=(h, w, 3), dtype=np.uint8)
            self.observation_space = gym.spaces.Dict(spaces)
        else:
            # 單影像
            h, w = self.size
            self.observation_space = gym.spaces.Box(low=0, high=255, shape=(h, w, 3), dtype=np.uint8)

    def observation(self, obs):
        img = obs.get("image", None) if isinstance(obs, dict) else obs
        if img is None:
            return obs
        if img.ndim == 3 and img.shape[-1] in (1, 3, 4):
            # HWC
            dst = cv2.resize(img[..., :3], self.size[::-1], interpolation=cv2.INTER_AREA)
        elif img.ndim == 3 and img.shape[0] in (1, 3, 4):
            # CHW -> HWC
            chw = img[:3, ...]
            dst = cv2.resize(np.transpose(chw, (1, 2, 0)), self.size[::-1], interpolation=cv2.INTER_AREA)
        else:
            # 非預期，直接返回
            return obs

        if isinstance(obs, dict):
            out = dict(obs)
            out["image"] = dst.astype(np.uint8)
            return out
        return dst.astype(np.uint8)
