"""
VLMRewardWrapper — add shaping from a Vision-Language Model (Ollama)
- Every `interval` env steps, ask a VLM to score tidiness (0..1) on the current image.
- Shaping = coeff * (score_now - score_ref). In preference mode, we score both frames and use sign(delta).
- Robust to failures: if VLM call fails, shaping=0 and training continues.

Requirements:
- A running Ollama with a vision model (e.g., qwen2.5-vl or qwen3-vl)
- Pillow installed for image encoding

Config knobs:
- mode: "score" | "preference"
- interval: evaluate every N steps (default 5)
- coeff: scaling of shaping term (default 0.5)
- prompt: override default prompt if desired
- host/model: override Ollama endpoint/model (or set env OLLAMA_HOST / OLLAMA_VLM)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import gymnasium as gym

from vlm.sync_api import score_image


@dataclass
class VLMWrapperConfig:
    mode: str = "score"          # or "preference"
    interval: int = 5
    coeff: float = 0.5
    prompt: Optional[str] = None
    host: Optional[str] = None
    model: Optional[str] = None
    clip_delta: float = 1.0       # clip |delta| to avoid spikes


class VLMRewardWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, cfg: Optional[VLMWrapperConfig] = None):
        super().__init__(env)
        self.cfg = cfg or VLMWrapperConfig()
        self._last_score: Optional[float] = None
        self._last_frame: Optional[np.ndarray] = None
        self._step_count: int = 0
        self._last_eval_step: int = -1

    # --------------------------------------------------------------
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_score = None
        self._last_frame = None
        self._step_count = 0
        self._last_eval_step = -1
        return obs, info

    # --------------------------------------------------------------
    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(action)
        self._step_count += 1
        shaping = 0.0
        cur_score = None
        prev_score = self._last_score

        if (self._step_count % max(1, self.cfg.interval)) == 0:
            frame = self._extract_image(obs)
            if frame is not None:
                # --- score current frame
                prompt = self.cfg.prompt or (
                    "You are a tidy desk judge. Score how tidy this tabletop arrangement looks. "
                    "Return a single number from 0 to 1 where 1=very tidy, 0=very messy. Output only the number."
                )
                cur_score = score_image(frame, prompt=prompt, host=self.cfg.host, model=self.cfg.model)
                if cur_score is not None:
                    if self.cfg.mode == "score":
                        delta = cur_score - (prev_score if prev_score is not None else cur_score)
                    else:  # preference-like: sign of improvement vs prev (fallback to score mode if no prev)
                        if prev_score is None:
                            delta = 0.0
                        else:
                            delta = np.sign(cur_score - prev_score)
                    # clip & scale
                    delta = float(np.clip(delta, -self.cfg.clip_delta, +self.cfg.clip_delta))
                    shaping = float(self.cfg.coeff * delta)
                    rew = float(rew + shaping)
                    self._last_score = float(cur_score)
                    self._last_frame = frame
                    self._last_eval_step = self._step_count

        # enrich info
        info = info or {}
        if cur_score is not None or prev_score is not None:
            info.update({
                "vlm_score": cur_score,
                "vlm_prev_score": prev_score,
                "vlm_delta": None if cur_score is None else (cur_score - (prev_score if prev_score is not None else cur_score)) if self.cfg.mode=="score" else (None if prev_score is None or cur_score is None else float(np.sign(cur_score - prev_score))),
                "vlm_shaping": shaping,
                "vlm_last_eval_step": self._last_eval_step,
            })
        return obs, rew, term, trunc, info

    # --------------------------------------------------------------
    def _extract_image(self, obs) -> Optional[np.ndarray]:
        # prefer obs dict
        if isinstance(obs, dict):
            img = obs.get("image")
            if img is not None:
                arr = np.asarray(img)
                # enforce uint8 RGB
                if arr.dtype != np.uint8:
                    arr = arr.astype(np.uint8)
                if arr.ndim == 2:
                    arr = np.stack([arr]*3, axis=-1)
                if arr.ndim == 3 and arr.shape[2] == 3:
                    return arr
                if arr.ndim == 3 and arr.shape[2] == 4:
                    return arr[:, :, :3]
        # fallback: try env.render if available
        try:
            frame = self.env.render()
            if frame is not None:
                arr = np.asarray(frame)
                if arr.dtype != np.uint8:
                    arr = arr.astype(np.uint8)
                if arr.ndim == 2:
                    arr = np.stack([arr]*3, axis=-1)
                if arr.ndim == 3 and arr.shape[2] >= 3:
                    return arr[:, :, :3]
        except Exception:
            pass
        return None


__all__ = ["VLMRewardWrapper", "VLMWrapperConfig"]
