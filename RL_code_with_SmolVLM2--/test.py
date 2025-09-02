"""
Smoke test: wire up XArm6 env + spawner (+ optional VLM wrapper) and run a few steps.
- Confirms that reset() spawns objects on the table and that step() returns penalty info.
- Does NOT train. Pure sanity check for your ROS2/Gazebo stack.

Usage (example):
    python scripts/smoke_test_env.py \
        --table-xmin -0.30 --table-xmax 0.30 --table-ymin 0.20 --table-ymax 0.80 --table-z 0.76 \
        --episodes 2 --steps 15 --mode sample_catalog

If you want VLM shaping enabled (requires a running Ollama with a vision model):
    python scripts/smoke_test_env.py --vlm 1

Notes:
- Replace the sample `file_path` below with real URDF/SDF files you have.
- If a `file_path` isn't found, the spawner will warn and skip that model.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import rclpy

# Adjust path if needed (assumes running from repo root)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from envs.xarm6_env import XArm6GymEnv, EnvConfig
from objects.spawner import SpawnerConfig, TableArea, ModelSpec
from wrappers.spawn_on_reset_wrapper import SpawnOnResetWrapper, SpawnOnResetConfig

# Optional VLM shaping
try:
    from wrappers.vlm_reward_wrapper import VLMRewardWrapper, VLMWrapperConfig
    VLM_OK = True
except Exception:
    VLM_OK = False


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--episodes', type=int, default=2)
    p.add_argument('--steps', type=int, default=20)
    p.add_argument('--seed', type=int, default=123)

    # table area
    p.add_argument('--table-xmin', type=float, default=-0.30)
    p.add_argument('--table-xmax', type=float, default=+0.30)
    p.add_argument('--table-ymin', type=float, default=+0.20)
    p.add_argument('--table-ymax', type=float, default=+0.80)
    p.add_argument('--table-z', type=float, default=0.76)

    # spawn mode
    p.add_argument('--mode', type=str, default='sample_catalog', choices=['use_specs','sample_catalog'])
    p.add_argument('--min-n', type=int, default=2)
    p.add_argument('--max-n', type=int, default=3)

    # VLM
    p.add_argument('--vlm', type=int, default=0, help='1 to enable VLM shaping (requires Ollama running)')
    p.add_argument('--vlm-interval', type=int, default=5)
    p.add_argument('--vlm-coeff', type=float, default=0.5)

    return p.parse_args()


def make_catalog() -> list[ModelSpec]:
    # TODO: replace these file paths with your actual models
    return [
        ModelSpec(name='beer', file_path='/home/m416-3090ti/.gazebo/models/beer/model-1_4.sdf', fmt='sdf', radius=0.035),
        ModelSpec(name='wood_cube_5cm',  file_path='/home/m416-3090ti/.gazebo/models/wood_cube_5cm/model.sdf',  fmt='sdf',  radius=0.045),
        ModelSpec(name='wood_cube_7_5cm',  file_path='/home/m416-3090ti/.gazebo/models/wood_cube_7_5cm/model.sdf', fmt='sdf', radius=0.030),
    ]


def main():
    args = parse_args()
    np.random.seed(args.seed)

    rclpy.init(args=None)

    # Base env (penalty-only safety is built-in)
    env = XArm6GymEnv(EnvConfig(
        robot_model='UF_ROBOT',
        gripper_attach_links=['left_finger','right_finger'],
    ))

    # Spawner wrapper
    sp_cfg = SpawnerConfig(
        table_area=TableArea(
            xmin=args.table_xmin, xmax=args.table_xmax,
            ymin=args.table_ymin, ymax=args.table_ymax,
            z=args.table_z,
        )
    )

    if args.mode == 'use_specs':
        specs = make_catalog()  # or a fixed subset
        sor_cfg = SpawnOnResetConfig(mode='use_specs', specs=specs, randomize_pose=True, seed=args.seed)
    else:
        catalog = make_catalog()
        sor_cfg = SpawnOnResetConfig(mode='sample_catalog', catalog=catalog,
                                     min_n=args.min_n, max_n=args.max_n,
                                     randomize_pose=True, seed=args.seed)

    env = SpawnOnResetWrapper(env, sp_cfg, sor_cfg)

    # Optional VLM shaping
    if args.vlm and VLM_OK:
        env = VLMRewardWrapper(env, VLMWrapperConfig(mode='score', interval=args.vlm_interval, coeff=args.vlm_coeff))
    elif args.vlm and not VLM_OK:
        print('[warn] wrappers.vlm_reward_wrapper not importable; running without VLM shaping')

    # Roll a few episodes
    for ep in range(args.episodes):
        obs, info = env.reset()
        spawned = info.get('spawned_models', [])
        print(f"[EP {ep}] spawned: {[m['name'] for m in spawned]} | table_z={info.get('table_z')} rect={info.get('table_rect')}")

        for t in range(args.steps):
            # random actions just to exercise the pipeline
            act = np.random.uniform(-1.0, 1.0, size=(7,)).astype(np.float32)
            obs, rew, term, trunc, info = env.step(act)
            print(f"  t={t:02d} reward={rew:+.3f} pen={info.get('penalty',0):.3f} "
                  f"Δ={info.get('penalty_deltas',{})} attached={info.get('attached',False)} "
                  f"safety={info.get('safety_score',None)}")
            if term or trunc:
                break

    env.close()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
