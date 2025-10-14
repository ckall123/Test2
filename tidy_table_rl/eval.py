import argparse
import os
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy

from xarm6_gym_env import XArm6Env
import utils


def make_env(node: Node, executor: MultiThreadedExecutor):
    cfg = utils.EnvConfig()
    env = XArm6Env(node, executor, cfg)
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str, help="Path to SB3 .zip model (e.g., sac_xarm6.zip)")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()

    if not os.path.isfile(args.model_path):
        raise FileNotFoundError(args.model_path)

    if not rclpy.ok():
        rclpy.init()

    node = Node("eval_sac_xarm6")
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        # Build env and wrap the same way as training
        base_env = make_env(node, executor)
        venv = DummyVecEnv([lambda: base_env])
        venv = VecTransposeImage(venv)

        # Load model and bind env
        model = SAC.load(args.model_path, device="cuda")
        model.set_env(venv)

        # Quick quantitative eval
        mean_r, std_r = evaluate_policy(model, venv, n_eval_episodes=args.episodes, deterministic=args.deterministic, render=False)
        print(f"[EVAL] {os.path.basename(args.model_path)}: mean_reward={mean_r:.3f} ± {std_r:.3f} over {args.episodes} eps")

        # One demo rollout (logs per-step); helps ensure arm ACTUALLY moves
        obs = venv.reset()
        done = [False]
        ep_r = 0.0
        step = 0
        while not done[0]:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, done, infos = venv.step(action)
            ep_r += float(reward[0])
            step += 1
            if step % 10 == 0:
                # Optional: print partial stats
                print(f"  step={step:03d} partial_return={ep_r:.3f}")
            # If your env adds reasons, surface them
            if done[0]:
                info = infos[0] if isinstance(infos, (list, tuple)) else infos
                reason = None
                if isinstance(info, dict):
                    reason = info.get('done_reason') or info.get('termination_reason')
                print(f"[DEMO] finished in {step} steps, return={ep_r:.3f}, reason={reason}")

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
