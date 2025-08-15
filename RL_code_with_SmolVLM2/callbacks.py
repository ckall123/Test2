import os, cv2, numpy as np
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.callbacks import BaseCallback

class TBCallback(BaseCallback):
    def __init__(self, log_dir="runs/xarm6_griptraj"):
        super().__init__(); self.w = SummaryWriter(log_dir); self.ep = 0
    def _on_step(self):
        # 必須實作以避免抽象類報錯；此處不做事，僅回傳 True
        return True
    def _on_rollout_end(self):
        if len(self.model.ep_info_buffer) > 0:
            last = self.model.ep_info_buffer[-1]
            if "r" in last: self.w.add_scalar("Episode/Reward", last["r"], self.ep)
            if "l" in last: self.w.add_scalar("Episode/Length", last["l"], self.ep)
            self.ep += 1
    def _on_training_end(self): self.w.close()

class ImageLogger(BaseCallback):
    def __init__(self, log_dir="runs/xarm6_griptraj", save_every=2000):
        super().__init__(); self.dir = os.path.join(log_dir, "samples"); os.makedirs(self.dir, exist_ok=True); self.k = save_every
    def _on_step(self):
        if self.num_timesteps % self.k == 0:
            env = self.training_env.envs[0] if hasattr(self.training_env, "envs") else self.training_env
            base = getattr(env, "unwrapped", env)
            img = getattr(base, "latest_image", None)
            if img is not None: cv2.imwrite(os.path.join(self.dir, f"step_{int(self.num_timesteps)}.jpg"), img)
        return True

class ActionHistogram(BaseCallback):
    def __init__(self, log_dir="runs/xarm6_griptraj", every=1000):
        super().__init__(); self.w = SummaryWriter(log_dir); self.k = every
    def _on_step(self):
        if self.num_timesteps % self.k == 0:
            actions = self.locals.get("actions", None)
            if actions is not None:
                a = np.array(actions)
                self.w.add_histogram("Actions/all", a, self.num_timesteps)
                if a.shape[-1] >= 7:
                    self.w.add_histogram("Actions/gripper", a[...,6], self.num_timesteps)
        return True
    def _on_training_end(self): self.w.close()

class MetricsLogger(BaseCallback):
    def __init__(self, log_dir="runs/xarm6_griptraj", every=500):
        super().__init__(); self.w = SummaryWriter(log_dir); self.k = every
    def _on_step(self):
        if self.num_timesteps % self.k == 0:
            env = self.training_env.envs[0] if hasattr(self.training_env, "envs") else self.training_env
            base = getattr(env, "unwrapped", env)
            # VLM 分數
            if hasattr(base, "_last_vlm_score"):
                self.w.add_scalar("Metrics/VLM_score", float(base._last_vlm_score), self.num_timesteps)
            # 幾何/等距
            if hasattr(base, "_layout_score"):
                try: self.w.add_scalar("Metrics/Layout", float(base._layout_score()), self.num_timesteps)
                except Exception: pass
            # 夾爪位置
            if hasattr(base, "gripper_angle"):
                self.w.add_scalar("Metrics/Gripper", float(base.gripper_angle), self.num_timesteps)
        return True
    def _on_training_end(self): self.w.close()