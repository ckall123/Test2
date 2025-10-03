# utils.py
import numpy as np
import cv2
import hashlib
from typing import Tuple, Dict
from reward import reward_alignment, reward_vlm, reward_combined
from camera import TopDownCamera
from attach_detach import attach_object, detach_object
from gripper_contact import in_contact
from vlm import VLMScorer


class EnvConfig:
    image_size = (128, 128)
    action_scale = 0.05
    max_steps = 100
    success_r_align = 0.85
    success_r_vlm = 0.7
    vlm_interval = 5
    attach_depth_threshold = 0.0002
    home_pose = [0.0] * 6  # 依照你的機器手臂修改


# ---- 圖像處理與觀測包裝 ----
def get_rgb(camera: TopDownCamera, cfg: EnvConfig) -> np.ndarray:
    image = camera.get_latest_frame()
    image_resized = cv2.resize(image, cfg.image_size)
    return image_resized

def is_black_image(image: np.ndarray, threshold=3) -> bool:
    return np.mean(image) < threshold

def pack_obs(rgb: np.ndarray, proprio: np.ndarray) -> Dict:
    return {
        'image': rgb,
        'proprio': proprio.astype(np.float32)
    }


# ---- 動作處理 ----
def action_to_targets(action: np.ndarray, current_joint: np.ndarray, cfg: EnvConfig) -> Tuple[np.ndarray, float]:
    delta = action[:-1] * cfg.action_scale
    target_joints = current_joint + delta
    gripper = float(np.clip(action[-1], 0.0, 1.0))
    return target_joints, gripper


# ---- 獎勵計算 ----
class VLMClock:
    def __init__(self, interval=5):
        self.interval = interval
        self.counter = 0
        self.last_score = 0.0

    def step(self):
        self.counter += 1

    def should_run(self) -> bool:
        return self.counter % self.interval == 0

    def update(self, score: float):
        self.last_score = score

    def get(self) -> float:
        return self.last_score


def compute_reward(image: np.ndarray, cfg: EnvConfig, vlm_clock: VLMClock, instruction="align objects in a row") -> Tuple[float, Dict]:
    r_align = reward_alignment(image)
    if vlm_clock.should_run():
        r_vlm = reward_vlm(image, instruction)
        vlm_clock.update(r_vlm)
    else:
        r_vlm = vlm_clock.get()

    r_total = 0.7 * r_align + 0.3 * r_vlm
    return r_total, {"r_align": r_align, "r_vlm": r_vlm, "r_total": r_total}


# ---- 成功判定 & info 組裝 ----
def check_success(r_info: Dict, cfg: EnvConfig) -> bool:
    return r_info["r_align"] > cfg.success_r_align and r_info["r_vlm"] > cfg.success_r_vlm

def make_info(**kwargs):
    return kwargs


# ---- 夾持操作封裝 ----
def try_attach(target: str) -> bool:
    if in_contact(target):
        attach_object(target)
        return True
    return False

def try_detach(target: str) -> bool:
    detach_object(target)
    return True
