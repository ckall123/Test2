import rclpy
import numpy as np
import torch
from stable_baselines3 import SAC
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor
from torch.utils.tensorboard import SummaryWriter
from xarm6_env import XArm6GymEnv

SEED = 42

def main():
    # 統一由最外層負責 init/shutdown
    if not rclpy.ok():
        rclpy.init()

    writer = None
    env = None
    try:
        writer = SummaryWriter("runs/xarm6_eval")

        # 如果 VLM 很吃資源，eval 可適度降頻，例如 5 或 10
        env = XArm6GymEnv(max_steps=200, vlm_interval=0)

        # 建議順序：TimeLimit 先包，再用 Monitor 最外層記錄
        env = Monitor(TimeLimit(env, max_episode_steps=200))

        # 若你想在沒有 GPU 的機器上也能跑，改成 auto：
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SAC.load("checkpoints_slim/best.zip", env=env, device=device)

        rs, layout, vlm = [], [], []

        for ep in range(10):
            obs, _ = env.reset(seed=SEED + ep)
            done = False
            trunc = False
            R = 0.0
            Ls = []
            Vs = []

            while not (done or trunc):
                a, _ = model.predict(obs, deterministic=True)
                obs, r, done, trunc, info = env.step(a)
                R += float(r)

                # ✅ 建議：之後把 layout / vlm 分數放進 info["layout_score"], info["vlm_score"]
                # 目前先沿用你的私有介面存取方式：
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
        # 溫柔收尾：有就關，沒有就略過
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
        if env is not None:
            try:
                env.close()   # ⚠️ 確保你的 env.close() 不會呼叫 rclpy.shutdown()
            except Exception:
                pass
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()
