import rclpy
import numpy as np
import torch
from stable_baselines3 import SAC
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor
from torch.utils.tensorboard import SummaryWriter
from envs.xarm6_env import XArm6GymEnv

SEED = 42

def main():
    if not rclpy.ok():
        rclpy.init()

    writer = None
    env = None
    try:
        writer = SummaryWriter("runs/xarm6_eval")
        env = XArm6GymEnv(max_steps=200, vlm_interval=0)
        env = Monitor(TimeLimit(env, max_episode_steps=200))

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SAC.load("checkpoints_slim/best.zip", env=env, device=device)

        rs, layout, vlm = [], [], []
        for ep in range(10):
            obs, _ = env.reset(seed=SEED + ep)
            done = False
            trunc = False
            R = 0.0
            Ls, Vs = [], []

            while not (done or trunc):
                a, _ = model.predict(obs, deterministic=True)
                obs, r, done, trunc, info = env.step(a)
                R += float(r)

                if hasattr(env, "_layout_score"):
                    try:
                        Ls.append(float(env._layout_score()))
                    except Exception:
                        pass
                Vs.append(float(getattr(env, "_last_vlm_score", 0.0)))

            rs.append(R)
            layout.append(np.median(Ls) if Ls else 0.0)
            vlm.append(np.median(Vs) if Vs else 0.0)

            writer.add_scalar("Eval/Reward", R, ep)
            writer.add_scalar("Eval/LayoutMedian", layout[-1], ep)
            writer.add_scalar("Eval/VLMMedian", vlm[-1], ep)

        print({
            "reward_avg": float(np.mean(rs)) if rs else 0.0,
            "layout_med": float(np.mean(layout)) if layout else 0.0,
            "vlm_med": float(np.mean(vlm)) if vlm else 0.0,
        })

    finally:
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()