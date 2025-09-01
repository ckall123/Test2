# =========================
# FILE: wrappers/add_state_key_wrapper.py
# =========================
from __future__ import annotations
from typing import Any, Dict

try:
    import gymnasium as gym
    from gymnasium.spaces import Box, Dict as SpaceDict
except Exception:
    import gym
    from gym.spaces import Box, Dict as SpaceDict

import numpy as np


class AddStateKeyWrapper(gym.ObservationWrapper):
    """
    將 obs 中的低維資訊合併為 `state` 鍵，並同步設定 env.latest_image。
    state = [tcp_pose(7), joint_pos(6), gripper_pos(1), traj_error(1), ee_low(1)] → 共 16 維
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        # 推斷 state 維度
        tcp_dim = int(env.observation_space["tcp_pose"].shape[0])
        jnt_dim = int(env.observation_space["joint_pos"].shape[0])
        grp_dim = int(env.observation_space["gripper_pos"].shape[0])
        flags_dim = 2  # traj_error, ee_low
        state_dim = tcp_dim + jnt_dim + grp_dim + flags_dim

        new_space: Dict[str, Any] = dict(env.observation_space.spaces)
        new_space["state"] = Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32)
        self.observation_space = SpaceDict(new_space)

    def observation(self, obs):
        tcp = np.asarray(obs.get("tcp_pose", np.zeros(7, np.float32)), dtype=np.float32)
        jnt = np.asarray(obs.get("joint_pos", np.zeros(6, np.float32)), dtype=np.float32)
        grp = np.asarray(obs.get("gripper_pos", np.zeros(1, np.float32)), dtype=np.float32)
        flags = np.array(
            [float(obs.get("traj_error", 0)), float(obs.get("ee_low", 0))],
            dtype=np.float32,
        )
        state = np.concatenate([tcp, jnt, grp, flags], axis=0).astype(np.float32)

        out = dict(obs)
        out["state"] = state

        # 曝露最新影像給需要的 wrapper
        try:
            self.env.latest_image = out.get("image", None)
        except Exception:
            pass
        return out
