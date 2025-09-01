# wrappers/add_state_key_wrapper.py
from __future__ import annotations
try:
    import gymnasium as gym
except Exception:
    import gym

import numpy as np


class AddStateKeyWrapper(gym.ObservationWrapper):
    """將 obs 中的低維資訊合併為 `state` 鍵，並同步設定 env.latest_image。"""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        # 嘗試擴充 observation_space：若空間不可變，就在 runtime 動態賦值即可
        try:
            from gym.spaces import Box
            # 估計長度：tcp_pose(7) + joint_pos(6) + gripper_pos(1) + flags(2)
            self.observation_space["state"] = Box(low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32)
        except Exception:
            pass

    def observation(self, obs):
        tcp = np.asarray(obs.get("tcp_pose", np.zeros(7, np.float32)), dtype=np.float32)
        jnt = np.asarray(obs.get("joint_pos", np.zeros(6, np.float32)), dtype=np.float32)
        grp = np.asarray(obs.get("gripper_pos", np.zeros(1, np.float32)), dtype=np.float32)
        flags = np.array([
            float(obs.get("traj_error", 0.0)),
            float(obs.get("ee_low", 0.0)),
        ], dtype=np.float32)
        state = np.concatenate([tcp, jnt, grp, flags], axis=0).astype(np.float32)
        try:
            obs["state"] = state
        except Exception:
            # 若 obs 是不可變結構，可以返回新的 dict
            obs = dict(obs)
            obs["state"] = state
        # 曝露最新影像給外部（ImageLogger / VLM wrapper 會用到）
        try:
            self.env.latest_image = obs.get("image", None)
        except Exception:
            pass
        return obs