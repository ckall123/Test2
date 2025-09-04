"""
Clean training script for TidyCubeEnv using Gymnasium + Stable-Baselines3 v2
- Zero hacks/wrappers, env keeps Gymnasium API (reset -> (obs, info); step -> 5-tuple)
- Dict observation handled via MultiInputPolicy
- Only the image key ("pixels") is transposed from HWC->CHW via VecTransposeImage(key=...)

Prereqs (install once):
    pip install -U "stable-baselines3[extra]>=2.0" gymnasium shimmy

Note: Using CPU device by default (your CUDA driver warning suggests mismatch).
Switch to device="auto" or "cuda" after fixing drivers.
"""
from __future__ import annotations
import os
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecMonitor

# Import your environment (must output a Dict obs with keys: "pixels" (HWC uint8), "agent_pos" (1D float))
from env.tidy_cube_env import TidyCubeEnv


def make_env():
    # Keep your env unchanged; it should follow Gymnasium's API
    # reset() -> (obs, info)
    # step(action) -> (obs, reward, terminated, truncated, info)
    return TidyCubeEnv()


def sanity_check_single_env() -> None:
    """Quick check to fail-fast if the env contract is broken."""
    env0 = TidyCubeEnv()
    obs, info = env0.reset()
    assert isinstance(obs, dict), f"Expected dict obs, got {type(obs)}"
    assert "pixels" in obs and "agent_pos" in obs, f"Obs keys missing, got: {list(obs.keys())}"
    # pixels must be HWC uint8 (e.g., 64x64x3)
    assert hasattr(obs["pixels"], "ndim") and obs["pixels"].ndim == 3, "obs['pixels'] must be HWC image"
    # agent_pos must be 1D vector
    assert hasattr(obs["agent_pos"], "ndim") and obs["agent_pos"].ndim == 1, "obs['agent_pos'] must be 1D vector"
    env0.close()


def main():
    # Optional: quick contract check
    sanity_check_single_env()

    # Build vectorized env (single instance to avoid ROS node name conflicts)
    env = DummyVecEnv([make_env])

    # Transpose only the image branch of the Dict observation (HWC -> CHW)
    env = VecTransposeImage(env, key="pixels")

    # Monitor training stats (episode length/return)
    env = VecMonitor(env)

    # SAC with MultiInputPolicy for Dict observations
    model = SAC(
        policy="MultiInputPolicy",
        env=env,
        device="cpu",            # use "auto" or "cuda" after fixing drivers
        buffer_size=200_000,      # reduce memory usage vs huge default buffers
        batch_size=256,
        learning_starts=1_000,
        train_freq=(1, "step"),
        gradient_steps=1,
        verbose=1,
        tensorboard_log="./tb_sac/",
    )

    total_timesteps = 50_000
    model.learn(total_timesteps=total_timesteps)

    os.makedirs("./models", exist_ok=True)
    model.save("./models/sac_tidy_cube_clean")

    env.close()


if __name__ == "__main__":
    main()
