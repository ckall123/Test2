"""
AddStateKeyWrapper — robust v2
--------------------------------
Purpose: ensure observation has a unified vector key "state" for MultiInputPolicy,
without assuming specific keys exist (no hardcoded 'tcp_pose').

• Concatenates any of the configurable source keys that are present, e.g.:
  ['tcp_pose', 'ee_pose', 'joint_positions', 'joint_velocities', 'gripper_opening', 'attached_flag']
• Missing keys are simply skipped (no error). Order is preserved.
• If none of the keys exist, 'state' is still provided with shape=(0,) (empty vector).
• Observation *dict* keeps original entries (image, etc.) and adds 'state'.
• Observation space is updated at init from env.observation_space; if shapes are
  not available, we fall back to first reset to infer lengths.

This wrapper is intentionally conservative: it does not change or remove existing
keys so that other components depending on raw keys continue to work.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any
import numpy as np
import gymnasium as gym
from gymnasium import spaces


@dataclass
class AddStateKeyConfig:
    # Candidate keys to concatenate (in this order if present)
    source_keys: List[str] = field(default_factory=lambda: [
        'tcp_pose',          # e.g., 6 or 7 elems
        'ee_pose',           # alternative to tcp_pose if available
        'joint_positions',   # arm joints
        'joint_velocities',
        'gripper_opening',   # scalar
        'attached_flag',     # scalar {0,1}
    ])
    dtype: np.dtype = np.float32
    key_name: str = 'state'


class AddStateKeyWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, cfg: AddStateKeyConfig | None = None):
        super().__init__(env)
        self.cfg = cfg or AddStateKeyConfig()
        self._lengths: Dict[str, int] = {}
        self._initialized = False

        # Try to build observation_space immediately (preferred by SB3)
        self.observation_space = self._build_observation_space_from(self.env.observation_space)

    # ------------------------------------------------------------------
    def _build_observation_space_from(self, base_space: spaces.Space) -> spaces.Space:
        """Return a new Dict space with an added 'state' Box. If base is not a Dict,
        we wrap it into {'image': base} for convenience.
        """
        if isinstance(base_space, spaces.Dict):
            space_dict = dict(base_space.spaces)
        else:
            # Wrap non-dict observation as image
            space_dict = {'image': base_space}

        # Determine per-key lengths from spaces if available
        total = 0
        lengths: Dict[str, int] = {}
        for k in self.cfg.source_keys:
            sp = space_dict.get(k)
            if isinstance(sp, spaces.Box):
                # flatten length (empty if unknown)
                n = int(np.prod(sp.shape))
            else:
                n = 0
            lengths[k] = n
            total += n
        # Save as hint; may be refined on first reset()
        self._lengths = lengths

        # Create/override the 'state' Box
        # Allow empty vector (shape=(0,)) when no keys present
        low = np.zeros((total,), dtype=self.cfg.dtype)
        high = np.zeros((total,), dtype=self.cfg.dtype)
        state_space = spaces.Box(low=low, high=high, dtype=self.cfg.dtype)
        space_dict[self.cfg.key_name] = state_space
        return spaces.Dict(space_dict)

    # ------------------------------------------------------------------
    def observation(self, obs: Any) -> Dict[str, Any]:
        """Add a 'state' vector constructed from available keys; keep originals."""
        if not isinstance(obs, dict):
            # Wrap to dict with image
            obs = {'image': obs}

        # Optionally refine lengths on the very first actual observation
        if not self._initialized:
            total = 0
            for k in self.cfg.source_keys:
                v = obs.get(k, None)
                n = int(np.prod(np.asarray(v).shape)) if v is not None else self._lengths.get(k, 0)
                self._lengths[k] = n
                total += n
            # Update observation_space.state if needed
            try:
                space_dict = dict(self.observation_space.spaces) if isinstance(self.observation_space, spaces.Dict) else {}
                low = np.zeros((total,), dtype=self.cfg.dtype)
                high = np.zeros((total,), dtype=self.cfg.dtype)
                space_dict[self.cfg.key_name] = spaces.Box(low=low, high=high, dtype=self.cfg.dtype)
                self.observation_space = spaces.Dict(space_dict)
            except Exception:
                # If SB3 already bound shapes, we still return consistent shapes at runtime
                pass
            self._initialized = True

        # Build the state vector
        parts: List[np.ndarray] = []
        for k in self.cfg.source_keys:
            v = obs.get(k, None)
            if v is None:
                n = self._lengths.get(k, 0)
                if n > 0:
                    parts.append(np.zeros((n,), dtype=self.cfg.dtype))
                continue
            arr = np.asarray(v, dtype=self.cfg.dtype).reshape(-1)
            parts.append(arr)
        if parts:
            state = np.concatenate(parts, axis=0).astype(self.cfg.dtype, copy=False)
        else:
            state = np.zeros((0,), dtype=self.cfg.dtype)

        # Attach
        obs[self.cfg.key_name] = state
        return obs


__all__ = ["AddStateKeyWrapper", "AddStateKeyConfig"]
