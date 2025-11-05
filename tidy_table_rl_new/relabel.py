import numpy as np
from typing import List, Dict, Any
import torch
from reward import RewardModel, preprocess_image


def relabel_transitions(
    replay: List[Dict[str, Any]],
    model: RewardModel,
    mode: str = "absolute",
    reward_range: tuple = (-1, 1),
    device: str = "cpu",
) -> List[Dict[str, Any]]:
    """
    支援 reward 重新標注，支援三種模式：
    - "absolute" / "abs": r = r(next_image)
    - "delta" / "diff":  r = r(next_image) - r(image)
    並做 z-score 標準化 + clip。
    """
    images = []
    next_images = []
    for t in replay:
        images.append(preprocess_image(t["image"]))
        next_images.append(preprocess_image(t["next_image"]))

    images = torch.stack(images).to(device)
    next_images = torch.stack(next_images).to(device)

    model.eval()
    with torch.no_grad():
        r_curr = model(images)
        r_next = model(next_images)

    # --- 模式選擇 ---
    mode_lower = mode.lower()
    if mode_lower in ("absolute", "abs"):
        raw_rewards = r_next.cpu().numpy()
    elif mode_lower in ("delta", "diff"):
        raw_rewards = (r_next - r_curr).cpu().numpy()
    else:
        raise ValueError("mode must be 'absolute'/'abs' or 'delta'/'diff'")

    # --- 標準化 ---
    mean = np.mean(raw_rewards)
    std = np.std(raw_rewards) + 1e-6
    rewards = (raw_rewards - mean) / std

    # --- clip 到指定範圍 ---
    low, high = reward_range
    rewards = np.clip(rewards, low, high)

    # --- 回寫 replay buffer ---
    for t, r in zip(replay, rewards):
        t["reward"] = float(r)

    return replay
