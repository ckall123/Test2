"""
SB3 SAC trainer (v3) — minimal, VSCode 一鍵執行；已修正 rclpy.init() 位置
- Uses your XArm6GymEnv (safety penalty inside Env)
- SpawnOnResetWrapper controls Gazebo spawns each reset
- Optional VLM shaping via VLMRewardWrapper (Ollama qwen2/3-vl)
- Keeps MultiInputPolicy + (optional) SmolVLAExtractor

Run:
  EP_MAX_STEPS=120 STEPS=200000 USE_VLM=1 \
  python scripts/train.py
"""
from __future__ import annotations

import os
import sys
import numpy as np
import torch
import rclpy  # <<< NEW: must init before creating envs

from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import SAC
from stable_baselines3.sac.policies import MultiInputPolicy

# Ensure repo root on sys.path when run from VSCode
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from envs.xarm6_env import XArm6GymEnv, EnvConfig
from objects.spawner import SpawnerConfig, TableArea, ModelSpec
from wrappers.spawn_on_reset_wrapper import SpawnOnResetWrapper, SpawnOnResetConfig

# Optional wrappers (存在就用，沒有也不會壞)
try:
    from wrappers.add_state_key_wrapper import AddStateKeyWrapper
    HAS_ADD_STATE = True
except Exception:
    HAS_ADD_STATE = False
try:
    from wrappers.image_resize_wrapper import ImageResizeWrapper
    HAS_IMG_RESIZE = True
except Exception:
    HAS_IMG_RESIZE = False

# VLM shaping（新介面以 config 物件傳入）
try:
    from wrappers.vlm_reward_wrapper import VLMRewardWrapper, VLMWrapperConfig
    HAS_VLM = True
except Exception:
    HAS_VLM = False

# 你的 SmolVLA 特徵抽取器（若存在就用）
FX_CLASS = None
try:
    from models.feature_extractor import SmolVLAExtractor as FX_CLASS
except Exception:
    FX_CLASS = None

SEED = int(os.environ.get("SEED", "42"))
LOGDIR = os.environ.get("LOGDIR", "runs/xarm6_sac")
CKPT_DIR = os.path.join(LOGDIR, "ckpts")
BEST_DIR = os.path.join(LOGDIR, "best")
os.makedirs(LOGDIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(BEST_DIR, exist_ok=True)

# torch 優化
torch.set_num_threads(1)
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# ===== knobs =====
KNOBS = {
    # VLM / 影像
    "VLM_INTERVAL": int(os.environ.get("VLM_INTERVAL", "10")),
    "VLM_PROMPT": os.environ.get("VLM_PROMPT", (
        "請嚴格評估這張桌面是否整齊：以『直線對齊、等間距、無重疊、皆在桌面內』為標準，"
        "給 0~1 分，只輸出數字。"
    )),
    "OLLAMA_VLM": os.environ.get("OLLAMA_VLM", "qwen2.5-vl"),
    "IMAGE_SIZE": int(os.environ.get("IMAGE_SIZE", "96")),

    # 桌面範圍（Spawner 用）
    "TABLE_XMIN": float(os.environ.get("TABLE_XMIN", "-0.30")),
    "TABLE_XMAX": float(os.environ.get("TABLE_XMAX", "+0.30")),
    "TABLE_YMIN": float(os.environ.get("TABLE_YMIN", "+0.20")),
    "TABLE_YMAX": float(os.environ.get("TABLE_YMAX", "+0.80")),
    "TABLE_Z": float(os.environ.get("TABLE_Z", "0.76")),
}

# === 你要把這裡換成真實可用的 URDF/SDF 路徑 ===
def make_catalog() -> list[ModelSpec]:
    base = os.environ.get("MODEL_DIR", "/home/user/models")
    return [
        ModelSpec(name="wood_cube_5cm",  file_path=os.path.join(base, "wood_cube_5cm.urdf"),   fmt="urdf", radius=0.025),
        ModelSpec(name="wood_cube_7_5cm",file_path=os.path.join(base, "wood_cube_7_5cm.urdf"),fmt="urdf", radius=0.0375),
        ModelSpec(name="beer",           file_path=os.path.join(base, "beer.sdf"),           fmt="sdf",  radius=0.030),
    ]


def make_env(max_steps: int, use_vlm: bool = True, mode: str = "sample_catalog"):
    def _thunk():
        # --- 保險：每個子環境建立前都嘗試 init（若已 init 會丟例外，我們吞掉） ---
        try:
            rclpy.init(args=None)
        except Exception:
            pass

        # 基礎環境（安全扣分內建）
        env = XArm6GymEnv(EnvConfig(
            robot_model='UF_ROBOT',
            gripper_attach_links=['left_finger','right_finger'],
        ))

        # 每回合 spawn 物件
        sp_cfg = SpawnerConfig(table_area=TableArea(
            xmin=KNOBS["TABLE_XMIN"], xmax=KNOBS["TABLE_XMAX"],
            ymin=KNOBS["TABLE_YMIN"], ymax=KNOBS["TABLE_YMAX"],
            z=KNOBS["TABLE_Z"],
        ))
        catalog = make_catalog()
        if mode == "use_specs":
            sor_cfg = SpawnOnResetConfig(mode='use_specs', specs=catalog, randomize_pose=True, seed=SEED)
        else:
            sor_cfg = SpawnOnResetConfig(mode='sample_catalog', catalog=catalog,
                                         min_n=2, max_n=3, randomize_pose=True, seed=SEED)
        env = SpawnOnResetWrapper(env, sp_cfg, sor_cfg)

        # 先補 state / 原圖供其他 wrapper 使用
        if HAS_ADD_STATE:
            env = AddStateKeyWrapper(env)

        # 讓 VLM 看原圖（放在 resize 前面）
        if use_vlm and HAS_VLM:
            env = VLMRewardWrapper(env, VLMWrapperConfig(
                mode='score', interval=KNOBS["VLM_INTERVAL"], coeff=0.4,
                model=KNOBS["OLLAMA_VLM"], prompt=KNOBS["VLM_PROMPT"],
            ))

        # 再縮圖（如果你有這個 wrapper）
        if HAS_IMG_RESIZE:
            env = ImageResizeWrapper(env, size=(KNOBS["IMAGE_SIZE"], KNOBS["IMAGE_SIZE"]))

        env = Monitor(env)
        env = TimeLimit(env, max_episode_steps=max_steps)
        return env
    return _thunk


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # --- 主要 init（在建立 DummyVecEnv / 環境之前）---
    try:
        rclpy.init(args=None)
    except Exception:
        pass

    max_steps = int(os.environ.get("EP_MAX_STEPS", "120"))
    n_envs = int(os.environ.get("N_ENVS", "1"))
    use_vlm = bool(int(os.environ.get("USE_VLM", "1")))

    env = DummyVecEnv([make_env(max_steps, use_vlm=use_vlm, mode=os.environ.get("SPAWN_MODE", "sample_catalog")) for _ in range(n_envs)])
    eval_env = DummyVecEnv([make_env(max_steps, use_vlm=use_vlm, mode=os.environ.get("SPAWN_MODE", "sample_catalog"))])

    policy_kwargs = {}
    if FX_CLASS is not None:
        policy_kwargs = dict(
            features_extractor_class=FX_CLASS,
            features_extractor_kwargs=dict(
                model_name=os.environ.get("SMOLVLA_MODEL", "lerobot/smolvla_base"),
                out_dim=512,
                freeze=True,
                prompt=KNOBS["VLM_PROMPT"],
                device=None,
            ),
            net_arch=dict(pi=[512, 256], qf=[512, 256]),
        )

    agent = SAC(
        policy=MultiInputPolicy,
        env=env,
        learning_rate=float(os.environ.get("LR", "3e-4")),
        buffer_size=int(os.environ.get("BUFFER", "25000")),
        batch_size=int(os.environ.get("BATCH", "256")),
        train_freq=int(os.environ.get("TRAIN_FREQ", "64")),
        gradient_steps=int(os.environ.get("GRAD_STEPS", "64")),
        tau=float(os.environ.get("TAU", "0.02")),
        gamma=float(os.environ.get("GAMMA", "0.98")),
        policy_kwargs=policy_kwargs,
        tensorboard_log=LOGDIR,
        verbose=1,
        seed=SEED,
        device="auto",
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=BEST_DIR,
        log_path=os.path.join(LOGDIR, "eval"),
        eval_freq=int(os.environ.get("EVAL_FREQ", "10000")),
        n_eval_episodes=int(os.environ.get("N_EVAL_EP", "3")),
        deterministic=True,
        render=False,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=int(os.environ.get("CKPT_FREQ", "20000")),
        save_path=CKPT_DIR,
        name_prefix="sac_xarm6",
        save_replay_buffer=False,
    )

    total_steps = int(os.environ.get("STEPS", "300000"))
    agent.learn(total_timesteps=total_steps, callback=[eval_cb, ckpt_cb], log_interval=10, progress_bar=False)

    agent.save(os.path.join(CKPT_DIR, "sac_xarm6_last"))
    env.close(); eval_env.close()

    # --- clean shutdown ---
    try:
        rclpy.shutdown()
    except Exception:
        pass


if __name__ == "__main__":
    main()
