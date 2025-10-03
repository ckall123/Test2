import os
import argparse
import datetime

import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from xarm6_gym_env import XArm6TidyEnv


def make_env(seed=None, **kwargs):
    def _init():
        env = XArm6TidyEnv(**kwargs)
        env = Monitor(env)
        return env
    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--total-steps', type=int, default=200_000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--logdir', type=str, default='./logs')
    parser.add_argument('--save-freq', type=int, default=10_000)
    parser.add_argument('--eval-freq', type=int, default=10_000)
    parser.add_argument('--vlm-end-only', action='store_true')
    parser.add_argument('--vlm-every-n', type=int, default=10)
    parser.add_argument('--w-align', type=float, default=0.7)
    parser.add_argument('--w-vlm', type=float, default=0.3)
    args = parser.parse_args()

    run_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(args.logdir, run_name)
    os.makedirs(log_path, exist_ok=True)

    env_fn = make_env(
        seed=args.seed,
        end_only_vlm=args.vlm_end_only,
        vlm_every_n=args.vlm_every_n,
        reward_weights=(args.w_align, args.w_vlm),
    )
    vec_env = DummyVecEnv([env_fn])

    model = SAC("CnnPolicy", vec_env,
                verbose=1,
                tensorboard_log=log_path,
                seed=args.seed,
                learning_rate=3e-4,
                batch_size=64,
                buffer_size=100_000,
                train_freq=1,
                gradient_steps=1,
                gamma=0.99,
                tau=0.005,
                device='auto')

    checkpoint_callback = CheckpointCallback(save_freq=args.save_freq,
                                             save_path=log_path,
                                             name_prefix="sac_model")

    eval_env = DummyVecEnv([make_env(seed=args.seed)])
    eval_callback = EvalCallback(eval_env,
                                 best_model_save_path=log_path,
                                 log_path=log_path,
                                 eval_freq=args.eval_freq,
                                 deterministic=True,
                                 render=False)

    model.learn(total_timesteps=args.total_steps,
                callback=[checkpoint_callback, eval_callback])

    model.save(os.path.join(log_path, "final_model"))
    print("✅ 訓練完成，模型已儲存！")


if __name__ == '__main__':
    main()
