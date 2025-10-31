#!/usr/bin/env python3
"""
train.py
訓練三件套（與 eval 一致）：
1) DummyVecEnv（ROS/Gazebo 友善）
2) 影像轉置 HWC→CHW（只轉 'image'；MultiInputPolicy 需要）
3) VecNormalize（標準化回報；保留 obs 原樣，避免動到影像）
並放大網路：policy_kwargs = {"net_arch": [512, 512, 256]}
"""

import os
from typing import Callable, List, Dict, Any

import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnvWrapper, VecNormalize
from gymnasium import spaces

from xarm6_gym_env import XArm6Env
from pose_tracker import PoseTracker, load_target_names
from spawner import Spawner
import utils


# =========================
# 影像轉置（Dict 專用）
# =========================
class TransposeImageDict(VecEnvWrapper):
    """
    將 DictObs 中的 'image' 從 HWC→CHW，保持其他鍵不變。
    - 和 eval 一致（MultiInputPolicy 期望 CHW）
    - 不影響 'state'，利於後續 VecNormalize 僅標準化回報（norm_reward）
    """
    def __init__(self, venv):
        assert isinstance(venv.observation_space, spaces.Dict), "需要 Dict 觀測：{'image', 'state'}"
        super().__init__(venv)
        obs_space: spaces.Dict = venv.observation_space
        assert "image" in obs_space.spaces and "state" in obs_space.spaces, "Dict 必須含 'image' 與 'state'"

        img_space: spaces.Box = obs_space.spaces["image"]
        assert img_space.shape[-1] == 3, "image 必須為 HWC（最後一維=3）"
        h, w, c = img_space.shape
        transposed_img = spaces.Box(low=0, high=255, shape=(c, h, w), dtype=img_space.dtype)

        self.observation_space = spaces.Dict({
            "image": transposed_img,
            "state": obs_space.spaces["state"],
        })

    def reset(self) -> Dict[str, Any]:
        obs = self.venv.reset()
        obs["image"] = obs["image"].transpose(0, 3, 1, 2)  # (N,H,W,C)->(N,C,H,W)
        return obs

    def step_wait(self):
        obs, rew, term, trunc, infos = self.venv.step_wait()
        obs["image"] = obs["image"].transpose(0, 3, 1, 2)
        return obs, rew, term, trunc, infos


# =========================
# 場景啟動（只在訓練一開始做一次）
# =========================
def bootstrap_scene(node: Node, executor: SingleThreadedExecutor, desired_count: int = 3) -> List[str]:
    """
    - 若 config.json 無 target_objects → spawn desired_count 個
    - 若有清單，但 Gazebo 場上不足 → 補齊到 desired_count
    - 回傳最新的目標名清單（env 的 round-robin 用）
    """
    names = load_target_names()
    spawner = Spawner(node=node, executor=executor)
    tracker = PoseTracker(node)

    if not names:
        spawner.spawn_random_objects(count=desired_count)
        return load_target_names()

    states = tracker.get_object_states(names)
    present = {s["name"] for s in states}
    need = max(0, desired_count - len(present))
    if need > 0:
        spawner.spawn_random_objects(count=need)
        return load_target_names()
    return names


# =========================
# VecEnv 建立
# =========================
def make_env(node: Node, executor: SingleThreadedExecutor, cfg: utils.EnvConfig) -> Callable[[], Monitor]:
    """工廠方法：建立單一環境並用 Monitor 記錄。"""
    def _init():
        env = XArm6Env(node=node, executor=executor, cfg=cfg)
        return Monitor(env)
    return _init


def build_vec_env(node: Node, executor: SingleThreadedExecutor, cfg: utils.EnvConfig):
    """
    DummyVecEnv(1) → TransposeImageDict(HWC→CHW) → VecNormalize(norm_reward=True)
    - 我們僅標準化回報（norm_obs=False），避免對影像做不必要的數值變換。
    """
    venv = DummyVecEnv([make_env(node, executor, cfg)])
    venv = TransposeImageDict(venv)
    venv = VecNormalize(venv, norm_obs=False, norm_reward=True, clip_reward=10.0)
    return venv


# =========================
# Main: 訓練
# =========================
def main():
    # ---- ROS 啟動 ----
    rclpy.init()
    executor = SingleThreadedExecutor()
    node = Node("train_node")
    executor.add_node(node)

    # ---- 場景一次性啟動（保留 soft reset 流程）----
    targets = bootstrap_scene(node, executor, desired_count=3)
    node.get_logger().info(f"[bootstrap] targets: {targets}")

    # ---- Config / VecEnv ----
    cfg = utils.EnvConfig()
    venv = build_vec_env(node, executor, cfg)

    # ---- Policy 與網路放大 ----
    policy = "MultiInputPolicy"
    policy_kwargs = dict(net_arch=[512, 512, 256])  # 影像 backbone 用預設 CNN，頭部放大

    # ---- PPO ----
    logdir = os.path.join("runs", "ppo_xarm6")
    os.makedirs(logdir, exist_ok=True)
    model = PPO(
        policy,
        venv,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=logdir,
        n_steps=2048,
        batch_size=256,
        gae_lambda=0.95,
        gamma=0.99,
        n_epochs=10,
        learning_rate=3e-4,
        clip_range=0.2,
        device="auto",
    )

    # ---- 訓練 ----
    total_timesteps = 300_000  # 依你的時間/資源調整
    model.learn(total_timesteps=total_timesteps, progress_bar=True)

    # ---- 保存模型與 VecNormalize 統計 ----
    save_dir = "models"
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, "ppo_xarm6")
    norm_path = os.path.join(save_dir, "vecnorm.pkl")
    model.save(model_path)
    venv.save(norm_path)  # VecNormalize 的 save/load

    node.get_logger().info(f"[save] model -> {model_path}.zip")
    node.get_logger().info(f"[save] vecnormalize -> {norm_path}")

    # ---- 清理 ----
    venv.close()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
