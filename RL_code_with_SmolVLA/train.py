import numpy as np
import rclpy
rclpy.init()

from xarm6_env2 import XArm6GymEnv
from buffers.image_buffer import ImageBuffer
from buffers.preference_dataset import PreferenceDataset
from buffers.replay_buffer import ReplayBuffer
from callbacks import TensorboardCallback

from stable_baselines3 import SAC
from stable_baselines3.sac.policies import MultiInputPolicy
from models.smolvla_extractor import SmolVLAExtractor

# from vlm_interface import get_vlm_preference  未來 VLM import

# === 設定 ===
TOTAL_EPISODES = 10
MAX_STEPS_PER_EPISODE = 100
VLM_INTERVAL = 20  # 每 K 步進行一次 VLM 偏好判斷

# === 初始化 ===
env = XArm6GymEnv()

policy_kwargs = dict(
    # features_extractor_class=SmolVLAExtractor,
    # features_extractor_kwargs=dict(out_dim=512),
)

image_buffer = env.image_buffer
preference_dataset = PreferenceDataset()
replay_buffer = ReplayBuffer()
callback = TensorboardCallback()

agent = SAC(
    MultiInputPolicy,
    env,
    # policy_kwargs=policy_kwargs,
    buffer_size=10_000,                 # ← 1e6 改 1e4
    # batch_size=256,
    # learning_starts=1_000,
    replay_buffer_kwargs=dict(optimize_memory_usage=True),  # ← 省一半記憶體
    verbose=1,
    device="cuda",
)

# === 訓練主迴圈 ===
for ep in range(TOTAL_EPISODES):
    obs, _ = env.reset()
    total_reward = 0
    for step in range(MAX_STEPS_PER_EPISODE):
        # action = env.action_space.sample()  # 暫時使用隨機策略  ||  action = agent.predict(obs)  ||
        action, _ = agent.predict(obs, deterministic=False)
        next_obs, reward, done, trunc, info = env.step(action)

        replay_buffer.add(obs, action, reward, next_obs, done)
        obs = next_obs
        total_reward += reward

        # 每 K 步從 image buffer 抽圖送去 VLM 評分（此處用 dummy）
        if step > 0 and step % VLM_INTERVAL == 0:
            pairs = image_buffer.sample_pairs(1)
            for a, b in pairs:
                y = np.random.choice([0, 1])  # dummy 偏好（未接 VLM） 未來換掉這行 改成 VLM   e.g. : y = get_vlm_preference(a.image, b.image, prompt="哪張桌面比較整齊？")
                preference_dataset.add(a.image, b.image, y)

        if done or trunc:
            break
                
    callback.log_episode(total_reward, step + 1)

# === 關閉 ===
callback.close()
env.close()  # 裡面已經包含 rclpy.shutdown()
