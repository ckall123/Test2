# =========================
# FILE: eval.py
# =========================
import os
import numpy as np
import torch

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor

from envs.xarm6_env import XArm6GymEnv
from reward.reward import XArmReward
from wrappers.spawn_on_reset_wrapper import SpawnOnResetWrapper
from wrappers.add_state_key_wrapper import AddStateKeyWrapper
from wrappers.image_resize_wrapper import ImageResizeWrapper
from wrappers.vlm_reward_wrapper import VLMRewardWrapper

SEED = int(os.environ.get("SEED", "42"))

def make_env(max_steps: int):
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
            arm_step_rad=0.20,
            arm_limit_margin=0.05,
            arm_time_sec=0.25,
            grip_min=0.0,
            grip_max=0.8552,
            grip_step=0.08,
            grip_time_sec=0.25,
            joint_weights=[1.0, 1.0, 0.8, 0.6, 0.6, 0.6],
            vlm_interval=int(os.environ.get("VLM_INTERVAL", "15")),
            vlm_prompt=os.environ.get("VLM_PROMPT", "請以『整齊程度：直線對齊、等間距、無重疊、在桌面內』為準則打分。"),
            camera_topic=os.environ.get("CAMERA_TOPIC", "/camera/image_raw"),
            rewarder=rewarder,
        )

        env = SpawnOnResetWrapper(env)
        env = ImageResizeWrapper(env, size=(int(os.environ.get("IMAGE_SIZE", "96")), int(os.environ.get("IMAGE_SIZE", "96"))))
        env = AddStateKeyWrapper(env)
        env = VLMRewardWrapper(
            env,
            interval=int(os.environ.get("VLM_INTERVAL", "15")),
            coeff=0.0,  # 只記錄不影響 reward
            model=os.environ.get("OLLAMA_VLM", "qwen2.5vl"),
            prompt=os.environ.get("VLM_PROMPT", None),
        )
        env = Monitor(env)
        env = TimeLimit(env, max_episode_steps=max_steps)
        return env
    return _thunk


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    max_steps = int(os.environ.get("EP_MAX_STEPS", "100"))
    n_episodes = int(os.environ.get("N_EPS", "10"))
    model_path = os.environ.get("MODEL", "runs/xarm6_sac/best/best_model.zip")

    env = DummyVecEnv([make_env(max_steps)])
    model = SAC.load(model_path, device="auto")

    vlm_medians = []
    returns = []

    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        ep_rew = 0.0
        vlm_scores = []
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = env.step(action)
            ep_rew += float(reward)
            info0 = infos[0] if isinstance(infos, (list, tuple)) and len(infos) > 0 else infos
            if isinstance(info0, dict) and "vlm_score" in info0:
                vlm_scores.append(float(info0["vlm_score"]))
            done = bool(dones)
        returns.append(ep_rew)
        vlm_medians.append(np.median(vlm_scores) if len(vlm_scores) > 0 else 0.0)
        print(f"[Eval] Episode {ep+1}/{n_episodes}: return={ep_rew:.3f}, VLM-median={vlm_medians[-1]:.3f}")

    print(f"== Summary ==")
    print(f"AvgReturn: {np.mean(returns):.3f}  |  VLM-Median: {np.mean(vlm_medians):.3f}")

    env.close()


if __name__ == "__main__":
    main()
