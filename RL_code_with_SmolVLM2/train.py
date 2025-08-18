import os
import rclpy
import numpy as np
import torch

from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import SAC
from stable_baselines3.sac.policies import MultiInputPolicy

from xarm6_env import XArm6GymEnv
from callbacks import TBCallback, ImageLogger, ActionHistogram, MetricsLogger
from models.feature_extractor import SmolVLM2Extractor

# ==== 基本設定 ====
SEED = int(os.environ.get("SEED", "42"))
LOGDIR = os.environ.get("LOGDIR", "runs/xarm6_train")
CKPT_DIR = os.environ.get("CKPT_DIR", "checkpoints")
BEST_DIR = os.environ.get("BEST_DIR", "checkpoints_best")

os.makedirs(LOGDIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(BEST_DIR, exist_ok=True)

# 限制 CPU threads，避免和 ROS 抢
torch.set_num_threads(1)
# PyTorch 2.x 可選：在 GPU 上提速；沒有也不影響
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
    # 各關節權重
    "JOINT_WEIGHTS": [1.0, 1.0, 0.8, 0.6, 0.6, 0.6],
    # VLM 與相機
    "VLM_INTERVAL": int(os.environ.get("VLM_INTERVAL", "15")),  # 訓練建議 10~20
    "VLM_PROMPT": "桌面物品整齊排列、等間距、邊緣對齊。",
    "CAMERA_TOPIC": "/camera/image_raw",
    # 特徵抽取器
    "USE_SMOLVLM2": True,
}

def make_env(knobs: dict, max_steps=200,
             arm_traj_topic="/xarm6_traj_controller/joint_trajectory",
             grip_action_name="/xarm_gripper_traj_controller/follow_joint_trajectory",
             gripper_joint_name="drive_joint"):
    # 由最外層統一 init/shutdown；這裡只保守保護
    if not rclpy.ok():
        rclpy.init()

    env = XArm6GymEnv(
        max_steps=max_steps,
        vlm_interval=knobs["VLM_INTERVAL"],
        vlm_prompt=knobs["VLM_PROMPT"],
        camera_topic=knobs["CAMERA_TOPIC"],
        arm_traj_topic=arm_traj_topic,
        grip_action_name=grip_action_name,
        gripper_joint_name=gripper_joint_name,
        # 旋鈕注入
        arm_step_rad=knobs["ARM_STEP_RAD"],
        arm_limit_margin=knobs["ARM_LIMIT_MARGIN"],
        arm_time_sec=knobs["ARM_TIME_SEC"],
        grip_min=knobs["GRIP_MIN"],
        grip_max=knobs["GRIP_MAX"],
        grip_step=knobs["GRIP_STEP"],
        grip_time_sec=knobs["GRIP_TIME_SEC"],
        joint_weights=knobs["JOINT_WEIGHTS"],
    )
    # 包裝順序：先 TimeLimit，再 Monitor 最外層
    env = Monitor(TimeLimit(env, max_episode_steps=max_steps))
    return env

def main():
    # 統一由最外層負責 ROS2 初始化和關閉
    if not rclpy.ok():
        rclpy.init()

    # 全域種子：SB3、Gym、Numpy、PyTorch
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    env = None
    eval_env = None
    try:
        # ---- 建立訓練環境 ----
        # 注意：ROS/Gazebo 典型只能單環境，這裡仍用 DummyVecEnv 方便 EvalCallback 配合
        def _make_train():
            e = make_env(KNOBS, max_steps=200)
            # Gymnasium reset seed
            e.reset(seed=SEED)
            return e
        env = DummyVecEnv([_make_train])

        # ---- 建立評估環境（較慢，但存最佳模型用）----
        # 如果你有 headless 模式可以關掉 ROS I/O，這裡可以 eval_env=headless=True
        def _make_eval():
            e = make_env(KNOBS, max_steps=200)
            e.reset(seed=SEED + 999)
            return e
        eval_env = DummyVecEnv([_make_eval])

        # ---- Policy kwargs ----
        if KNOBS["USE_SMOLVLM2"]:
            policy_kwargs = dict(
                features_extractor_class=SmolVLM2Extractor,
                features_extractor_kwargs=dict(
                    out_dim=512,
                    model_name="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
                    freeze=True,
                    prompt=KNOBS["VLM_PROMPT"],
                ),
            )
        else:
            policy_kwargs = dict()  # 用內建特徵抽取

        device = "cuda" if torch.cuda.is_available() else "cpu"

        agent = SAC(
            MultiInputPolicy,
            env,
            policy_kwargs=policy_kwargs,
            buffer_size=100_000,
            batch_size=256,
            learning_starts=2_000,
            train_freq=1,
            gradient_steps=1,   # 若訓練太慢可加大，例如 4
            tau=0.01,
            gamma=0.99,
            learning_rate=3e-4,
            verbose=1,
            device=device,
            tensorboard_log=LOGDIR,
            seed=SEED,
        )

        # ---- 回呼：TensorBoard/影像/動作分佈/自訂指標 ----
        tb_cb   = TBCallback(log_dir=LOGDIR)
        img_cb  = ImageLogger(log_dir=LOGDIR, save_every=2000)
        act_cb  = ActionHistogram(log_dir=LOGDIR, every=1000)
        metr_cb = MetricsLogger(log_dir=LOGDIR, every=500)

        # ---- 回呼：最佳模型 & 週期性 checkpoint ----
        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=BEST_DIR,
            log_path=os.path.join(LOGDIR, "eval"),
            eval_freq=10_000,            # 每 N timesteps 評估一次
            n_eval_episodes=3,
            deterministic=True,
            render=False,
        )
        ckpt_cb = CheckpointCallback(
            save_freq=20_000,
            save_path=CKPT_DIR,
            name_prefix="sac_xarm6",
            save_replay_buffer=False,    # 顯存/磁碟吃緊時先關
        )

        # ---- 開練 ----
        total_steps = int(os.environ.get("STEPS", "300000"))
        agent.learn(
            total_timesteps=total_steps,
            callback=[tb_cb, img_cb, act_cb, metr_cb, eval_cb, ckpt_cb],
            log_interval=10,
            progress_bar=False,
        )

        # 收尾：保存最後模型
        agent.save(os.path.join(CKPT_DIR, "sac_xarm6_last"))

    finally:
        # 安全收尾（有就關，沒有就算）
        try:
            if env is not None:
                env.close()  # 只 destroy_node，不要 shutdown
        except Exception:
            pass
        try:
            if eval_env is not None:
                eval_env.close()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()
