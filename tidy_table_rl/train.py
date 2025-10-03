# train.py
import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from xarm6_gym_env import XArm6Env
import utils


def make_env():
    cfg = utils.EnvConfig()
    env = XArm6Env(cfg)
    check_env(env, warn=True)
    return env


def train():
    env = make_env()
    model = SAC("MultiInputPolicy", env, verbose=1, tensorboard_log="./sac_xarm6_tensorboard")

    total_timesteps = 100_000
    model.learn(total_timesteps=total_timesteps)

    model.save("sac_xarm6")
    env.close()


if __name__ == '__main__':
    train()
