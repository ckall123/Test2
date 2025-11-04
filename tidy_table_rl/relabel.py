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
    支援 reward 重新標注，支援兩種模式：
    - "absolute": r = r(next_image)
    - "delta": r = r(next_image) - r(image)
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

    if mode == "absolute":
        raw_rewards = r_next.cpu().numpy()
    elif mode == "delta":
        raw_rewards = (r_next - r_curr).cpu().numpy()
    else:
        raise ValueError("mode must be 'absolute' or 'delta'")

    # Z-score 標準化
    mean = np.mean(raw_rewards)
    std = np.std(raw_rewards) + 1e-6
    rewards = (raw_rewards - mean) / std

    # Clip 到指定範圍
    low, high = reward_range
    rewards = np.clip(rewards, low, high)

    # 更新 replay buffer 中的 reward
    for t, r in zip(replay, rewards):
        t["reward"] = float(r)

    return replay
