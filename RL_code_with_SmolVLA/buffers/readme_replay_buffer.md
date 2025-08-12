🧠 模型內容說明：
add(obs, action, reward, next_obs, done)
加一筆 transition（就像你在 step() 執行後得到的）

sample_batch(batch_size)
隨機抽一批做 RL 訓練用（如 DQN、SAC、PPO 等）

update_rewards(new_rewards)
給一個新的 reward list，就能 relabel buffer 裡原有的 reward