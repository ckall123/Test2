#!/usr/bin/env python3
import rclpy
from rclpy.executors import SingleThreadedExecutor
import torch
import random
from reward import RewardModel, compute_bt_loss, make_transforms, relabel_transitions
from xarm6_gym_env import XArm6Env, XArmEnvConfig
from utils import sample_image_pairs, ask_vlm_preference
from stable_baselines3 import SAC

# ✅ 初始化 ROS2
rclpy.init()
node = rclpy.create_node('xarm6_train')
executor = SingleThreadedExecutor()
executor.add_node(node)

# ✅ 初始化環境與模型
env = XArm6Env(node, executor, XArmEnvConfig())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rpsi = RewardModel().to(device)

# ✅ 初始化 SAC Agent（使用 Stable-Baselines3）
# 若 obs 是 Dict("state", "image") 結構，可改 MultiInputPolicy
agent = SAC("MlpPolicy", env, verbose=1, device=device, learning_rate=3e-4, buffer_size=100000)

# ✅ 初始化 buffer
replay_buffer = []  # B
image_buffer = []   # I
pref_dataset = []   # D

# ✅ Hyperparams
N = 3000         # rollout 步數
M = 50           # 每輪偏好 pair 數
reward_scale = 0.1

for cycle in range(100):
    print(f"\n=== Cycle {cycle} 開始 ===")

    # 1️⃣ Rollout：互動 N 步
    obs, info = env.reset()
    for _ in range(N):
        action, _ = agent.predict(obs, deterministic=False)
        next_obs, _, done, truncated, next_info = env.step(action)

        transition = {
            'state': obs['state'],
            'action': action,
            'next_state': next_obs['state'],
            'image': info['image'],
            'next_image': info['next_image'],
            'reward': 0.0  # placeholder，之後會 relabel
        }
        replay_buffer.append(transition)
        image_buffer.append(info['next_image'])

        if done or truncated:
            obs, info = env.reset()
        else:
            obs, info = next_obs, next_info

    # 2️⃣ Preference：抽 M 對問 VLM
    pairs = sample_image_pairs(image_buffer, M)
    for imgA, imgB in pairs:
        y = ask_vlm_preference(imgA, imgB)
        if y in [0, 1]:
            pref_dataset.append((imgA, imgB, y))

    # 3️⃣ Train rψ (Reward Model)
    optimizer = torch.optim.Adam(rpsi.parameters(), lr=1e-4)
    transform = make_transforms()
    rpsi.train()

    for _ in range(1000):
        batch = random.sample(pref_dataset, min(32, len(pref_dataset)))
        imgs0, imgs1, labels = zip(*batch)
        x0 = torch.stack([transform(img) for img in imgs0]).to(device)
        x1 = torch.stack([transform(img) for img in imgs1]).to(device)
        y = torch.tensor(labels, dtype=torch.float32).to(device)

        r0 = rpsi(x0)
        r1 = rpsi(x1)
        loss = compute_bt_loss(r0, r1, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 4️⃣ Relabel rewards
    relabel_transitions(replay_buffer, rpsi, device, diff_mode=True)

    # 5️⃣ Policy Update
    # 把 relabeled reward 寫入 SB3 replay buffer 並學習
    for t in replay_buffer:
        agent.replay_buffer.add(
            t['state'], t['next_state'],
            t['action'], t['reward'] * reward_scale,
            done=False
        )

    agent.learn(total_timesteps=len(replay_buffer), log_interval=10)
    print(f"✅ SAC Policy Updated (Cycle {cycle})")

# ✅ 收尾
agent.save("sac_xarm6_policy")
node.destroy_node()
rclpy.shutdown()
print("🎯 訓練完成，模型已儲存為 sac_xarm6_policy.zip")
