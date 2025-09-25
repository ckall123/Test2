#!/usr/bin/env python3
"""
train.py - 強化學習訓練主程式
使用 SAC 搭配 CnnPolicy，從 Gazebo 模擬環境學習桌面整理策略。
"""

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from env.xarm6_gym_env import XArm6TidyEnv  # 根據實際模組調整
import os


def make_env():
    def _init():
        env = XArm6TidyEnv(
            reward_type="mixed",     # "geometric" / "vlm" / "mixed"
            use_vlm_end=True,        # 是否使用 VLM 作為結尾 reward
            max_steps=60,            # 每個 episode 最多步數
            render_mode=None,        # 若需開啟 GUI 可設為 "human"
            seed=42                  # 固定亂數種子
        )
        return Monitor(env)
    return _init


def main():
    # 環境設定
    env = DummyVecEnv([make_env()])
    env = VecTransposeImage(env)

    # 模型與日誌路徑
    log_dir = "./logs"
    model_path = os.path.join(log_dir, "tidy_sac_model")

    # 訓練 callback：定期儲存模型與評估
    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=log_dir,
        name_prefix="sac_checkpoint"
    )

    # 初始化 SAC 模型
    model = SAC(
        policy="CnnPolicy",
        env=env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=3e-4,
        buffer_size=100_000,
        batch_size=64,
        train_freq=1,
        gradient_steps=1,
        gamma=0.99,
        tau=0.005,
        device="auto"
    )

    # 開始訓練
    total_steps = 200_000
    model.learn(
        total_timesteps=total_steps,
        callback=checkpoint_callback
    )

    # 儲存最終模型
    model.save(model_path)
    print(f"✅ 模型訓練完成，已儲存至：{model_path}")


if __name__ == "__main__":
    main()
