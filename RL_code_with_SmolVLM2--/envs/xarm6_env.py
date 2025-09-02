"""
XArm6 Gymnasium environment (v5)
- Removed safety shield (no action blocking). The agent learns via **penalties only**.
- Step computes contact counts **before & after** motion and penalizes **new** violations.
- Still reports safety_score and counts in info for logging/analysis.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from ._ros_helpers import RosHelpers, RosConfig

# Optional safety scoring
try:
    from reward.reward import safety_score_from_contacts
except Exception:  # pragma: no cover
    safety_score_from_contacts = None  # type: ignore


@dataclass
class EnvConfig:
    # Action scaling
    joint_step: float = 0.05
    gripper_step: float = 0.04

    # Joint limits
    joint_lower: List[float] = field(default_factory=lambda: [-3.14]*6)
    joint_upper: List[float] = field(default_factory=lambda: [ 3.14]*6)

    # Gripper range
    gripper_min: float = 0.0
    gripper_max: float = 0.40

    # Attach/detach thresholds
    attach_threshold: float = 0.02
    detach_threshold: float = 0.06
    attach_eps: float = 1e-3

    # Names for link attacher
    robot_model: str = "UF_ROBOT"
    gripper_attach_links: List[str] = field(default_factory=lambda: ["left_finger", "right_finger"])
    ignored_attach_models: List[str] = field(default_factory=lambda: ["world", "ground_plane", "ground", "floor", "table"])

    # Timing
    traj_duration: float = 1.0
    spin_timeout: float = 0.05

    # Penalty weights (per new violation in this step)
    pen_self_collision: float = 1.0
    pen_robot_table: float = 0.5
    pen_arm_obj_nongrip: float = 0.25


class XArm6GymEnv(gym.Env):
    metadata = {"render.modes": ["rgb_array"]}

    def __init__(self,
                 env_cfg: Optional[EnvConfig] = None,
                 ros_cfg: Optional[RosConfig] = None,
                 use_action: Optional[bool] = None):
        super().__init__()
        self.env_cfg = env_cfg or EnvConfig()
        if ros_cfg is None:
            ros_cfg = RosConfig()
        if use_action is not None:
            ros_cfg.use_action = bool(use_action)
        self.ros = RosHelpers(ros_cfg)

        # Internal state
        self._attached: bool = False
        self._attached_target: Optional[Tuple[str, str]] = None
        self._last_gripper: float = 0.0

        # Observation space
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8),
            "state": spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32),
        })

        # Action space
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)

        self.ros.wait_until_ready()
        self._last_gripper = self.ros.get_gripper_position()

        # rolling previous violation counts (for delta penalty)
        self._prev_counts = {"self_collision": 0, "robot_table": 0, "arm_obj_nongrip": 0}

    # --------------------------------------------------------------
    def _add_tcp_pose_to_obs(self, obs: dict) -> dict:
        """
        查 TF: link_base -> link_tcp，將 6D tcp_pose 丟進 obs（若查不到就略過）。
        你目前的 frames：parent='link_base'、child='link_tcp'
        """
        try:
            pose6 = self.ros.get_link_pose('link_base', 'link_tcp', rpy=True)
        except Exception:
            pose6 = None
        if pose6 is not None:
            obs['tcp_pose'] = pose6.astype(np.float32)
        return obs

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._attached = False
        self._attached_target = None
        self._last_gripper = self.ros.get_gripper_position()
        self._prev_counts = {"self_collision": 0, "robot_table": 0, "arm_obj_nongrip": 0}
        obs = self._collect_obs()
        info = {"attached": self._attached}
        return obs, info

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        assert action.shape[0] == 7

        # pre-step contacts
        counts_pre = self._compute_counts()

        # Current joints + gripper
        cur_j = np.asarray(self.ros.get_joint_positions(), dtype=np.float32)
        cur_g = float(self.ros.get_gripper_position())

        # actions -> targets
        tgt_j = cur_j + action[:6] * self.env_cfg.joint_step
        tgt_g = cur_g + float(action[6]) * self.env_cfg.gripper_step

        # clamp
        tgt_j = np.clip(tgt_j, self.env_cfg.joint_lower, self.env_cfg.joint_upper)
        tgt_g = float(np.clip(tgt_g, self.env_cfg.gripper_min, self.env_cfg.gripper_max))

        # execute (no shield)
        self.ros.send_arm_traj(tgt_j.tolist(), duration=self.env_cfg.traj_duration)
        self.ros.send_gripper_traj(tgt_g, duration=self.env_cfg.traj_duration)
        self.ros.spin_once(self.env_cfg.spin_timeout)

        # attach/detach
        self._apply_gripper_and_attach_detach(cur_g, tgt_g)

        # post-step contacts & penalty
        counts_post = self._compute_counts()
        penalty, deltas = self._penalty_from_deltas(counts_pre, counts_post)

        # obs & info
        obs = self._collect_obs()
        info: Dict[str, Any] = {
            "attached": self._attached,
            "attached_target": self._attached_target,
            "gripper": tgt_g,
            "penalty": float(penalty),
            "penalty_deltas": deltas,
            **{f"safety_{k}": int(v) for k, v in counts_post.items()},
        }

        # safety score (optional)
        if safety_score_from_contacts is not None:
            pairs_now = self.ros.get_contact_pairs_latest(self.env_cfg.robot_model)
            safety, _ = safety_score_from_contacts(pairs_now, robot_model=self.env_cfg.robot_model,
                                                   gripper_links=self.env_cfg.gripper_attach_links,
                                                   table_models=["table","ground","ground_plane","world"],
                                                   attached_model=self._attached_target[0] if self._attached_target else None)
            info["safety_score"] = float(safety)

        reward = float(-penalty)  # penalty is positive -> negative reward
        terminated = False
        truncated = False
        return obs, reward, terminated, truncated, info

    # --------------------------------------------------------------
    def _compute_counts(self) -> Dict[str, int]:
        pairs = self.ros.get_contact_pairs_latest(self.env_cfg.robot_model)
        if safety_score_from_contacts is None:
            return dict(self._prev_counts)  # no change if module missing
        _, counts = safety_score_from_contacts(
            pairs,
            robot_model=self.env_cfg.robot_model,
            gripper_links=self.env_cfg.gripper_attach_links,
            table_models=["table","ground","ground_plane","world"],
            attached_model=self._attached_target[0] if self._attached_target else None,
        )
        # keep only keys we use
        out = {
            "self_collision": int(counts.get("self_collision", 0)),
            "robot_table": int(counts.get("robot_table", 0)),
            "arm_obj_nongrip": int(counts.get("arm_obj_nongrip", 0)),
        }
        return out

    def _penalty_from_deltas(self, pre: Dict[str, int], post: Dict[str, int]):
        # Penalize **new** violations only
        d_self = max(0, int(post.get("self_collision", 0)) - int(pre.get("self_collision", 0)))
        d_table = max(0, int(post.get("robot_table", 0)) - int(pre.get("robot_table", 0)))
        d_arm = max(0, int(post.get("arm_obj_nongrip", 0)) - int(pre.get("arm_obj_nongrip", 0)))
        pen = (
            self.env_cfg.pen_self_collision * d_self
            + self.env_cfg.pen_robot_table * d_table
            + self.env_cfg.pen_arm_obj_nongrip * d_arm
        )
        self._prev_counts = dict(post)
        return float(pen), {"self_collision": d_self, "robot_table": d_table, "arm_obj_nongrip": d_arm}

    # --------------------------------------------------------------
    def _apply_gripper_and_attach_detach(self, cur_g: float, tgt_g: float) -> None:
        # Gripper motion is already sent in step(); no-op here except attach/detach
        try_attach = (not self._attached) and (tgt_g < self.env_cfg.attach_threshold) 
        try_detach = self._attached and (tgt_g > self.env_cfg.detach_threshold)

        if try_attach:
            cand = self.ros.get_attach_candidate(
                robot_model=self.env_cfg.robot_model,
                gripper_links=self.env_cfg.gripper_attach_links,
                ignored_models=set(self.env_cfg.ignored_attach_models),
            )
            if cand is not None:
                other_model, other_link, grip_link = cand
                ok = self.ros.attach_link(other_model, other_link, self.env_cfg.robot_model, grip_link)
                if ok:
                    self._attached = True
                    self._attached_target = (other_model, other_link)
        elif try_detach and self._attached and self._attached_target is not None:
            other_model, other_link = self._attached_target
            grip_link = self.env_cfg.gripper_attach_links[0]
            ok = self.ros.detach_link(other_model, other_link, self.env_cfg.robot_model, grip_link)
            if ok:
                self._attached = False
                self._attached_target = None

        self._last_gripper = tgt_g

    # --------------------------------------------------------------
    def _collect_obs(self) -> Dict[str, Any]:
        img = self.ros.get_image(fill_if_none=True)
        j = np.asarray(self.ros.get_joint_positions(), dtype=np.float32)
        g = float(self.ros.get_gripper_position())
        attached = 1.0 if self._attached else 0.0
        state = np.concatenate([j, np.asarray([g, attached], dtype=np.float32)], axis=0)
        return {"image": img, "state": state}

    def render(self, mode: str = "rgb_array"):
        if mode != "rgb_array":
            raise NotImplementedError
        return self.ros.get_image(fill_if_none=True)

    def close(self):
        self.ros.close()
