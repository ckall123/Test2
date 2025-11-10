#!/usr/bin/env python3
"""
train.py — RL-VLM-F pipeline for xArm6 (Joint-centric)

流程：
1) 初始化 ROS2、環境、Agent(SAC)、回放/影像/偏好資料緩衝區、VLM 與 RewardModel。
2) 反覆進行迭代：
   - Rollout N 步：與環境互動、蒐集 transition 與圖片（含 episode/step 索引）。
   - 從 ImageBuffer 取樣 M 對影像，呼叫 VLMScorer 取得偏好標籤，累積 PreferenceDataset。
   - 用偏好資料訓練 RewardModel（Bradley–Terry pairwise loss）。
   - 用 RewardModel 對 replay transitions 做「diff」模式 relabel。
   - 將 relabeled transitions 寫入 SB3 replay buffer，僅執行「梯度更新」（不再收集新資料）。
3) 週期性儲存模型／紀錄。

注意：
- 僅在本檔案建立 ROS2 node/executor。
- SAC 使用 MultiInputPolicy（observation 是 Dict(image, state)）。
"""

import os
import random
from typing import List

import numpy as np
import torch
from torch import optim

import rclpy
from rclpy.executors import SingleThreadedExecutor

from stable_baselines3 import SAC

# === 本專案模組 ===
from xarm6_gym_env import XArm6Env, XArmEnvConfig
from buffers import ReplayBuffer, ImageBuffer, PreferenceDataset
from reward import RewardModel, pairwise_loss, preprocess_image
from relabel import relabel_transitions
from vlm import VLMScorer
from collision_object import CollisionObjectManager


# ========= 超參 =========
CYCLES: int = 100                 # 外圈迭代數
ROLLOUT_STEPS: int = 3000         # 每個 Cycle rollout 步數 N
PREF_PAIRS_PER_CYCLE: int = 50    # 每個 Cycle 送 VLM 的配對數 M
REWARD_SCALE: float = 0.1         # 寫入 SB3 buffer 前的縮放
BT_EPOCHS: int = 1000             # RewardModel 訓練步數（上限）
BT_BATCH: int = 32                # RewardModel 批次大小
LR_R: float = 1e-4                # RewardModel 學習率
SAVE_DIR: str = "runs"            # 輸出資料夾


# ========= 小工具 / 模組介面 =========
def _to_torch_batch(imgs: List[np.ndarray], device: torch.device) -> torch.Tensor:
    """將 numpy 影像 list 轉為批次 tensor（B,C,H,W），用 reward.preprocess_image。"""
    with torch.no_grad():
        tensors = [preprocess_image(img) for img in imgs]  # Tensor(C,H,W)
    return torch.stack(tensors, dim=0).to(device)          # (B,C,H,W)


def ask_vlm_preference(scorer: VLMScorer, img_a: np.ndarray, img_b: np.ndarray) -> int:
    """
    呼叫 VLMScorer 比較兩張圖的美感佈局。
    回傳：0 偏好 A、1 偏好 B、-1 無法分辨（會被過濾）。
    """
    rubric = (
        "- Items are aligned in straight rows or columns.\n"
        "- Empty space between items is balanced.\n"
        "- No excessive overlap or clutter.\n"
        "- Object orientations are consistent."
    )
    return scorer.compare(img_a, img_b, rubric)


def update_policy_from_replay(agent: SAC, replay_len: int, batch_size: int, logger) -> None:
    """
    僅以現有 replay buffer 做「梯度更新」，避免再次與環境互動。
    - 若 SB3 提供 agent.train(gradient_steps, batch_size)：直接使用（無新 rollouts）。
    - 否則退而求其次：把 gradient_steps 設為更新次數，調用 learn(total_timesteps=1)。
    """
    updates = max(replay_len // batch_size, 1)

    # 方案 A：直接呼叫 OffPolicyAlgorithm.train（不收集新資料）
    if hasattr(agent, "train"):
        agent.train(gradient_steps=updates, batch_size=batch_size)
        logger.info(f"✅ SAC Policy updated with {updates} gradient steps (no new rollouts)")
        return

    # 方案 B：版本受限時的折衷（僅 1 步 learn，主要吃 replay 的大量 gradient_steps）
    old_gs = getattr(agent, "gradient_steps", None)
    agent.gradient_steps = updates
    agent.learn(total_timesteps=1, reset_num_timesteps=False)
    if old_gs is not None:
        agent.gradient_steps = old_gs
    logger.info(f"✅ SAC Policy updated via 1-step learn with {updates} gradient steps")


def main():
    # ===== 1) ROS2 & 裝置 =====
    rclpy.init()
    node = rclpy.create_node("xarm6_train")
    executor = SingleThreadedExecutor()
    executor.add_node(node)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    node.get_logger().info(f"🚀 Device: {device}")

    CollisionObjectManager.default_setup(node, executor)

    try:
        # ===== 2) Env / Agent / Buffers / Models =====
        env_cfg = XArmEnvConfig()
        env = XArm6Env(node, executor, env_cfg)

        agent = SAC(
            policy="MultiInputPolicy",
            env=env,
            verbose=1,
            device=device,
            learning_rate=3e-4,
            buffer_size=100_000,
            train_freq=(1, "step"),
            gradient_steps=1,          # 實際更新步數在 update_policy_from_replay 動態控制
            batch_size=32,
            ent_coef="auto",
        )

        replay = ReplayBuffer()        # 自有 replay（僅供重標用）
        images = ImageBuffer()         # 影像緩衝
        prefs = PreferenceDataset()    # 偏好資料集

        vlm = VLMScorer(model="llama3.2-vision:11b")
        rpsi = RewardModel().to(device)
        optimizer_r = optim.Adam(rpsi.parameters(), lr=LR_R)

        global_step = 0
        episode_id = 0

        # ===== 3) Training Cycles =====
        for cycle in range(CYCLES):
            node.get_logger().info(f"\n=== Cycle {cycle} ===")

            # --- 3.1 Rollout ---
            obs, info = env.reset()
            step_in_ep = 0
            for _ in range(ROLLOUT_STEPS):
                action, _ = agent.predict(obs, deterministic=False)
                next_obs, _, terminated, truncated, next_info = env.step(action)
                done = bool(terminated or truncated)

                transition = {
                    "obs": obs,
                    "next_obs": next_obs,
                    "action": action,
                    "reward": 0.0,  # 佔位，稍後重標
                    "done": done,
                    "image": info["image"],
                    "next_image": next_info["next_image"],
                }
                replay.add(transition)
                images.add(episode_id, step_in_ep, next_info["next_image"])

                global_step += 1
                step_in_ep += 1
                if done:
                    episode_id += 1
                    obs, info = env.reset()
                    step_in_ep = 0
                else:
                    obs, info = next_obs, next_info

            # --- 3.2 偏好：抽 M 對影像問 VLM ---
            pairs = images.sample_pairs(PREF_PAIRS_PER_CYCLE)
            accepted = 0
            for imgA, imgB in pairs:
                y = ask_vlm_preference(vlm, imgA, imgB)
                if y in (0, 1):
                    prefs.add(imgA, imgB, y)
                    accepted += 1
            rej_rate = prefs.get_reject_rate()
            node.get_logger().info(f"🧮 偏好配對：取樣 {len(pairs)} 對，接受 {accepted}，拒絕率 {rej_rate:.2%}")

            # --- 3.3 訓練 RewardModel（Bradley–Terry）---
            rpsi.train()
            for _ in range(min(BT_EPOCHS, len(prefs))):
                batch = random.sample(prefs.get_all(), k=min(BT_BATCH, len(prefs)))
                imgs0, imgs1, labels = zip(*batch)

                x0 = _to_torch_batch(list(imgs0), device)
                x1 = _to_torch_batch(list(imgs1), device)
                y = torch.tensor(labels, dtype=torch.float32, device=device)

                r0 = rpsi(x0)               # (B,)
                r1 = rpsi(x1)               # (B,)

                # === 修正點 1：標籤方向對齊 BT 語意（0=A, 1=B → 1 代表 A>B） ===
                y_bt = 1.0 - y              # 0(A)→1, 1(B)→0
                loss = pairwise_loss(r0, r1, y_bt)

                optimizer_r.zero_grad()
                loss.backward()
                optimizer_r.step()
            rpsi.eval()
            node.get_logger().info("🎯 RewardModel 更新完成")

            # --- 3.4 用 RewardModel 重標（diff 模式）---
            relabel_transitions(
                replay=replay.data,
                model=rpsi,
                mode="diff",
                reward_range=(-1, 1),
                device=device,
            )
            node.get_logger().info("🔁 Replay rewards 已 relabel (diff)")

            # --- 3.5 寫入 SB3 ReplayBuffer & 僅做梯度更新 ---
            for t in replay.data:
                agent.replay_buffer.add(
                    obs=t["obs"],
                    next_obs=t["next_obs"],
                    action=t["action"],
                    reward=float(t["reward"]) * REWARD_SCALE,
                    done=t["done"],
                    infos={},
                )

            # === 修正點 2：只做梯度更新，不再觸發新 rollouts ===
            update_policy_from_replay(
                agent=agent,
                replay_len=len(replay),
                batch_size=agent.batch_size,
                logger=node.get_logger(),
            )

            # 本輪重標資料已用完，清除自有 replay（SB3 內部 buffer 保留）
            replay.clear()

        # ===== 4) 收尾與保存 =====
        os.makedirs(SAVE_DIR, exist_ok=True)
        agent.save(os.path.join(SAVE_DIR, "sac_xarm6_policy"))
        torch.save(rpsi.state_dict(), os.path.join(SAVE_DIR, "reward_model.pt"))
        node.get_logger().info("💾 已保存 policy 與 reward model")

    finally:
        try:
            env.close()
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()
        print("🎉 訓練完成")


if __name__ == "__main__":
    main()
