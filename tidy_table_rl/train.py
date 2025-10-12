import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env

from xarm6_gym_env import XArm6Env
import utils

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

from collision_object import setup_collision_objects

def make_env(node: Node, executor: MultiThreadedExecutor) -> XArm6Env:
    cfg = utils.EnvConfig()
    env = XArm6Env(node, executor, cfg)
    check_env(env, warn=True)
    return env

def train():
    if not rclpy.ok():
        rclpy.init()

    executor = MultiThreadedExecutor()
    node = Node("train_node")
    executor.add_node(node)

    # 初始化桌面碰撞物件（idempotent、安全）
    setup_collision_objects(node, executor)

    try:
        env = make_env(node, executor)
        model = SAC("MultiInputPolicy", env, verbose=1, tensorboard_log="./sac_xarm6_tensorboard")

        model.learn(total_timesteps=100_000)
        model.save("sac_xarm6")

        env.close()

    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    train()