import rclpy, numpy as np
from stable_baselines3 import SAC
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor
from torch.utils.tensorboard import SummaryWriter
from xarm6_env import XArm6GymEnv

SEED = 42

if __name__ == "__main__":
    if not rclpy.ok(): rclpy.init()
    writer = SummaryWriter("runs/xarm6_eval")
    env = XArm6GymEnv(max_steps=200, vlm_interval=0)
    env = TimeLimit(env, max_episode_steps=200)
    env = Monitor(env)
    model = SAC.load("checkpoints_slim/best.zip", env=env, device="cuda")
    rs, layout, vlm = [], [], []
    for ep in range(10):
        obs, _ = env.reset(seed=SEED+ep)
        done = trunc = False; R = 0.0; Ls = []; Vs = []
        while not (done or trunc):
            a, _ = model.predict(obs, deterministic=True)
            obs, r, done, trunc, info = env.step(a)
            R += float(r); Ls.append(float(env._layout_score())); Vs.append(float(getattr(env, "_last_vlm_score", 0.0)))
        rs.append(R); layout.append(np.median(Ls) if Ls else 0.0); vlm.append(np.median(Vs) if Vs else 0.0)
        writer.add_scalar("Eval/Reward", R, ep); writer.add_scalar("Eval/LayoutMedian", layout[-1], ep); writer.add_scalar("Eval/VLMMedian", vlm[-1], ep)
    print({"reward_avg": np.mean(rs), "layout_med": np.mean(layout), "vlm_med": np.mean(vlm)})
    writer.close(); env.close();
    if rclpy.ok(): rclpy.shutdown()