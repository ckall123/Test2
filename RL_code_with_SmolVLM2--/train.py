# =========================
# FILE: train.py
# =========================
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

# ==== 基本設定 ====
SEED = int(os.environ.get("SEED", "42"))
LOGDIR = os.environ.get("LOGDIR", "runs/xarm6_train")
CKPT_DIR = os.environ.get("CKPT_DIR", "checkpoints")
BEST_DIR = os.environ.get("BEST_DIR", "checkpoints_best")

os.makedirs(LOGDIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(BEST_DIR, exist_ok=True)

# 限制 CPU threads，避免和 ROS 搶
torch.set_num_threads(1)
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# ===== 可調旋鈕 =====
KNOBS = {
    # 手臂步長/安全/速度
    "ARM_STEP_RAD": 0.20,
    "ARM_LIMIT_MARGIN": 0.05,
    "ARM_TIME_SEC": 0.25,
    # 夾爪
    "GRIP_MIN": 0.0,
    "GRIP_MAX": 0.8552,
    "GRIP_STEP": 0.08,
    "GRIP_TIME_SEC": 0.25,
    # 各關節權重（for movement cost）
    "JOINT_WEIGHTS": [1.0, 1.0, 0.8, 0.6, 0.6, 0.6],
    # 相機 / VLM
    "VLM_INTERVAL": int(os.environ.get("VLM_INTERVAL", "15")),
    "VLM_PROMPT": "桌面物品整齊排列、等間距、邊緣對齊。",
    "CAMERA_TOPIC": os.environ.get("CAMERA_TOPIC", "/camera/image_raw"),
}


def make_env(knobs: dict, max_steps=200,
             arm_traj_topic="/xarm6_traj_controller/joint_trajectory",
             grip_action_name="/xarm_gripper_traj_controller/follow_joint_trajectory",
             gripper_joint_name="drive_joint"):
    # reward 物件：所有扣分/加分邏輯都在這裡
    rewarder = XArmReward(
        table_z=1.0,                 # 你的桌面在 1.0 m 附近
        safe_z_margin=0.03,
        close_gap_threshold=0.015,
        move_cost_coeff=0.01,
        violation_penalty=1.0,
        ee_low_penalty=0.5,
        step_penalty=0.01,
    )

    env = XArm6GymEnv(
        max_steps=max_steps,
        # ROS 控制 topic
        arm_traj_topic=arm_traj_topic,
        grip_action_name=grip_action_name,   # 會自動轉成 /joint_trajectory
        gripper_joint_name=gripper_joint_name,
        # 旋鈕
        arm_step_rad=knobs["ARM_STEP_RAD"],
        arm_limit_margin=knobs["ARM_LIMIT_MARGIN"],
        arm_time_sec=knobs["ARM_TIME_SEC"],
        grip_min=knobs["GRIP_MIN"],
        grip_max=knobs["GRIP_MAX"],
        grip_step=knobs["GRIP_STEP"],
        grip_time_sec=knobs["GRIP_TIME_SEC"],
        joint_weights=knobs["JOINT_WEIGHTS"],
        # 影像（供 VLM / CNN 用）
        vlm_interval=knobs["VLM_INTERVAL"],
        vlm_prompt=knobs["VLM_PROMPT"],
        camera_topic=knobs["CAMERA_TOPIC"],
        # 把 rewarder 丟給 env 使用（env 不做加減分，只回傳 rewarder 的結果）
        rewarder=rewarder,
    )
    env = SpawnOnResetWrapper(env,
                          object_names=("beer", "mug", "coke_can"),
                          count_range=(2, 4),
                          avoid_overlap_dist=0.10,
                          strict_unique=False)
    
    # 2) 將低維觀測拼成 state 鍵，並同步 latest_image
    env = AddStateKeyWrapper(env)

    # 3) 以 Qwen 2.5-VL 作為整潔評審，定期以 Δscore 形狀化獎勵
    env = VLMRewardWrapper(env,
                        model="qwen2.5vl",
                        interval=10,      # 每 10 步評一次
                        coeff=0.5)        # Δscore 係數，可 0.3~0.7 之間調

    env = Monitor(TimeLimit(env, max_episode_steps=max_steps))
    return env



def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    env = DummyVecEnv([lambda: make_env(KNOBS, max_steps=200)])
    eval_env = DummyVecEnv([lambda: make_env(KNOBS, max_steps=200)])

    # 這裡假設觀測含 image（由 env 提供），若你有自定義抽特徵器可在 policy_kwargs 指定
    policy_kwargs = dict()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    agent = SAC(
        MultiInputPolicy,
        env,
        policy_kwargs=policy_kwargs,
        buffer_size=100_000,
        batch_size=256,
        learning_starts=2_000,
        train_freq=1,
        gradient_steps=1,
        tau=0.01,
        gamma=0.99,
        learning_rate=3e-4,
        verbose=1,
        device=device,
        tensorboard_log=LOGDIR,
        seed=SEED,
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
    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
