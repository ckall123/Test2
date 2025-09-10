# train.py
import os, argparse
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.logger import configure

from env.xarm6_gym_env import XArm6TidyEnv

def make_env(args):
    env = XArm6TidyEnv(
        img_h=args.img_h, img_w=args.img_w,
        max_steps=args.horizon,
        step_dt=args.step_dt,
        use_vlm_end=not args.no_vlm_end,
        objects_n=args.objects,
    )
    return Monitor(env)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_name", default="sac_xarm6_tidy")
    p.add_argument("--total_steps", type=int, default=300_000)
    p.add_argument("--img_h", type=int, default=96)
    p.add_argument("--img_w", type=int, default=96)
    p.add_argument("--horizon", type=int, default=80)
    p.add_argument("--step_dt", type=float, default=0.8)
    p.add_argument("--no_vlm_end", action="store_true")
    p.add_argument("--objects", type=int, default=4)
    p.add_argument("--eval_freq", type=int, default=20_000)
    p.add_argument("--eval_eps", type=int, default=15)
    p.add_argument("--save_freq", type=int, default=50_000)
    args = p.parse_args()

    logdir = os.path.join("runs", args.run_name)
    os.makedirs(logdir, exist_ok=True)

    env = make_env(args)
    eval_env = make_env(args)

    model = SAC(
        policy="CnnPolicy",
        env=env,
        verbose=1,
        learning_rate=3e-4,
        buffer_size=200_000,
        batch_size=256,
        gamma=0.99,
        tau=0.01,
        train_freq=(1, "episode"),  # 回合末會加 VLM 分
        gradient_steps=1,
        tensorboard_log=logdir,
        device="auto",
    )

    model.set_logger(configure(logdir, ["stdout", "csv", "tensorboard"]))

    eval_cb = EvalCallback(
        eval_env, n_eval_episodes=args.eval_eps, eval_freq=args.eval_freq,
        deterministic=False, best_model_save_path=os.path.join(logdir, "best"),
        log_path=os.path.join(logdir, "eval"),
    )
    ckpt_cb = CheckpointCallback(
        save_freq=args.save_freq, save_path=os.path.join(logdir, "ckpt"),
        name_prefix="sac", save_replay_buffer=True, save_vecnormalize=True
    )

    model.learn(total_timesteps=args.total_steps, callback=[eval_cb, ckpt_cb])
    model.save(os.path.join(logdir, "final_model"))

if __name__ == "__main__":
    main()
