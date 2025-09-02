import os
import numpy as np
import torch

from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import SAC
from stable_baselines3.sac.policies import MultiInputPolicy

from envs.xarm6_env import XArm6GymEnv
from reward.reward import XArmReward
from wrappers.spawn_on_reset_wrapper import SpawnOnResetWrapper
from wrappers.add_state_key_wrapper import AddStateKeyWrapper
from wrappers.vlm_reward_wrapper import VLMRewardWrapper
from wrappers.image_resize_wrapper import ImageResizeWrapper
from models.feature_extractor import SmolVLAExtractor

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
    "ARM_STEP_RAD": 0.20,
    "ARM_LIMIT_MARGIN": 0.05,
    "ARM_TIME_SEC": 0.25,
    "GRIP_MIN": 0.0,
    "GRIP_MAX": 0.8552,
    "GRIP_STEP": 0.08,
    "GRIP_TIME_SEC": 0.25,
    "JOINT_WEIGHTS": [1.0, 1.0, 0.8, 0.6, 0.6, 0.6],
    "VLM_INTERVAL": int(os.environ.get("VLM_INTERVAL", "15")),
    "VLM_PROMPT": os.environ.get("VLM_PROMPT", "請嚴格評估這張桌面是否整齊：以『直線對齊、等間距、無重疊、皆在桌面內』為標準，給 0~1 分的 JSON。"),
    "CAMERA_TOPIC": os.environ.get("CAMERA_TOPIC", "/camera/image_raw"),
    "OLLAMA_VLM": os.environ.get("OLLAMA_VLM", "qwen2-vl"),  # 可改 qwen3-vl / qwen2.5-vl
    "IMAGE_SIZE": int(os.environ.get("IMAGE_SIZE", "96")),    # ReplayBuffer 用低解析
}

def make_env(max_steps: int, train: bool = True):
    def _thunk():
        arm_traj_topic = os.environ.get("ARM_TRAJ_TOPIC", "/xarm6_traj_controller/joint_trajectory")
        grip_action_name = os.environ.get("GRIP_ACTION_NAME", "/xarm_gripper_controller/follow_joint_trajectory")
        gripper_joint_name = os.environ.get("GRIPPER_JOINT", "drive_joint")

        rewarder = XArmReward(
            table_z=1.0,
            safe_z_margin=0.03,
            close_gap_threshold=0.015,
            move_cost_coeff=0.01,
            violation_penalty=1.0,
            ee_low_penalty=0.5,
            step_penalty=0.01,
        )

        env = XArm6GymEnv(
            max_steps=max_steps,
            arm_traj_topic=arm_traj_topic,
            grip_action_name=grip_action_name,
            gripper_joint_name=gripper_joint_name,
            arm_step_rad=KNOBS["ARM_STEP_RAD"],
            arm_limit_margin=KNOBS["ARM_LIMIT_MARGIN"],
            arm_time_sec=KNOBS["ARM_TIME_SEC"],
            grip_min=KNOBS["GRIP_MIN"],
            grip_max=KNOBS["GRIP_MAX"],
            grip_step=KNOBS["GRIP_STEP"],
            grip_time_sec=KNOBS["GRIP_TIME_SEC"],
            joint_weights=KNOBS["JOINT_WEIGHTS"],
            vlm_interval=KNOBS["VLM_INTERVAL"],
            vlm_prompt=KNOBS["VLM_PROMPT"],
            camera_topic=KNOBS["CAMERA_TOPIC"],
            rewarder=rewarder,
        )

        # ✅ Wrapper 順序：確保 VLM 看到 224×224 原圖
        env = SpawnOnResetWrapper(env)
        env = AddStateKeyWrapper(env)  # 先補 state 與 latest_image（供其他 wrapper 用）
        env = ImageResizeWrapper(env, size=(KNOBS["IMAGE_SIZE"], KNOBS["IMAGE_SIZE"]))  # 之後再縮圖
        env = VLMRewardWrapper(env, interval=KNOBS["VLM_INTERVAL"], coeff=0.4,
                               model=KNOBS["OLLAMA_VLM"], prompt=KNOBS["VLM_PROMPT"])  # shaping 僅此處
        env = Monitor(env)
        env = TimeLimit(env, max_episode_steps=max_steps)
        return env
    return _thunk


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    max_steps = int(os.environ.get("EP_MAX_STEPS", "100"))
    n_envs = int(os.environ.get("N_ENVS", "1"))

    env = DummyVecEnv([make_env(max_steps, train=True) for _ in range(n_envs)])
    eval_env = DummyVecEnv([make_env(max_steps, train=False)])

    policy_kwargs = dict(
        features_extractor_class=SmolVLAExtractor,
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
        learning_rate=3e-4,
        buffer_size=25_000,
        batch_size=256,
        train_freq=64,
        gradient_steps=64,
        tau=0.02,
        gamma=0.98,
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
        eval_freq=10_000,
        n_eval_episodes=3,
        deterministic=True,
        render=False,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=20_000,
        save_path=CKPT_DIR,
        name_prefix="sac_xarm6",
        save_replay_buffer=False,
    )

    total_steps = int(os.environ.get("STEPS", "300000"))
    agent.learn(
        total_timesteps=total_steps,
        callback=[eval_cb, ckpt_cb],
        log_interval=10,
        progress_bar=False,
    )

    agent.save(os.path.join(CKPT_DIR, "sac_xarm6_last"))
    env.close(); eval_env.close()


if __name__ == "__main__":
    main()