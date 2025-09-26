from env.xarm6_gym_env_copy import XArm6TidyEnv, EnvConfig

cfg = EnvConfig(gripper_mode="rule", use_vlm_end=False)  # 先不測 VLM
env = XArm6TidyEnv(cfg)

obs, info = env.reset()
print("✅ Reset OK, obs shape:", obs.shape)

for _ in range(5):
    action = env.action_space.sample()
    obs, r, term, trunc, info = env.step(action)
    print(f"🎯 Step reward: {r:.3f}, terminated={term}, truncated={trunc}")

env.close()
