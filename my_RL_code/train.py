import os
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from env.tidy_cube_env import TidyCubeEnv


def make_env():
    return TidyCubeEnv(render_mode=None)


def main():
    log_dir = "logs/sac_tidycube"
    os.makedirs(log_dir, exist_ok=True)

    env = DummyVecEnv([make_env])
    eval_env = DummyVecEnv([make_env])

    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=log_dir,
        name_prefix="sac_model"
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=5_000,
        deterministic=True,
        render=False,
    )

    model = SAC(
        policy="MultiInputPolicy",
        env=env,
        verbose=1,
        tensorboard_log=log_dir,
        device="auto"
    )

    model.learn(
        total_timesteps=100_000,
        callback=[checkpoint_callback, eval_callback]
    )

    model.save(os.path.join(log_dir, "final_model"))
    print("✅ Training complete. Model saved at:", os.path.join(log_dir, "final_model"))


if __name__ == "__main__":
    main()
