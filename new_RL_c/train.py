import os
import rclpy
import torch

from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import SAC
from stable_baselines3.sac.policies import MultiInputPolicy

from xarm6_env import XArm6GymEnv
from callbacks import TBCallback, ImageLogger, ActionHistogram, MetricsLogger
from models.feature_extractor import SmolVLM2Extractor

# 建議限制 CPU threads，避免跟 ROS callback 搶
torch.set_num_threads(1)

# ===== 可調旋鈕（你主要改這裡就好） =====
KNOBS = {
    # 手臂步長/安全/速度
    "ARM_STEP_RAD": 0.20,        # 每步最大關節位移（弧度）
    "ARM_LIMIT_MARGIN": 0.05,    # 接近關節極限時保留的邊界（弧度）
    "ARM_TIME_SEC": 0.25,        # arm JointTrajectory 的 time_from_start（秒）
    # 夾爪範圍/步長/速度
    "GRIP_MIN": 0.0,
    "GRIP_MAX": 0.8552,
    "GRIP_STEP": 0.08,           # 每步夾爪增量
    "GRIP_TIME_SEC": 0.25,       # gripper FollowJointTrajectory 的 time_from_start（秒）
    # 各關節權重（長度 6）→ 某些關節動更大或更小
    "JOINT_WEIGHTS": [1.0, 1.0, 0.8, 0.6, 0.6, 0.6],
    # VLM 與相機
    "VLM_INTERVAL": 20,          # 每幾步才詢問 VLM reward（省時間/顯存）
    "VLM_PROMPT": "桌面物品整齊排列、等間距、邊緣對齊。",
    "CAMERA_TOPIC": "/camera/image_raw",
    # 特徵抽取是否用 SmolVLM2（True）或 SB3 內建 CNN（False）
    "USE_SMOLVLM2": True,
}


def make_env(knobs: dict, max_steps=200,
             arm_traj_topic="/xarm6_traj_controller/joint_trajectory",
             grip_action_name="/xarm_gripper_traj_controller/follow_joint_trajectory",
             gripper_joint_name="drive_joint"):
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
    env = TimeLimit(env, max_episode_steps=max_steps)
    env = Monitor(env)
    return env


if __name__ == "__main__":
    logdir = os.environ.get("LOGDIR", "runs/xarm6_griptraj")
    os.makedirs(logdir, exist_ok=True)

    env = make_env(
        KNOBS,
        max_steps=200,
        arm_traj_topic="/xarm6_traj_controller/joint_trajectory",
        grip_action_name="/xarm_gripper_traj_controller/follow_joint_trajectory",
        gripper_joint_name="drive_joint",
    )

    if KNOBS["USE_SMOLVLM2"]:
        policy_kwargs = dict(
            features_extractor_class=SmolVLM2Extractor,
            features_extractor_kwargs=dict(
                out_dim=512,
                model_name="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
                freeze=True,
                prompt=KNOBS["VLM_PROMPT"],  # ← 這裡也可替換 prompt
            ),
        )
    else:
        policy_kwargs = {}

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
        verbose=1,
        device="cuda",
        tensorboard_log=logdir,
    )

    tb_cb   = TBCallback(log_dir=logdir)
    img_cb  = ImageLogger(log_dir=logdir, save_every=2000)
    act_cb  = ActionHistogram(log_dir=logdir, every=1000)
    metr_cb = MetricsLogger(log_dir=logdir, every=500)

    agent.learn(total_timesteps=300_000, callback=[tb_cb, img_cb, act_cb, metr_cb])

    env.close()
    if rclpy.ok():
        rclpy.shutdown()