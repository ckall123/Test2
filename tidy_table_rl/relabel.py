import numpy as np
from typing import List, Dict, Any, Union
import torch
from torch.cuda.amp import autocast
from reward import RewardModel, preprocess_image


def relabel_transitions(
    replay: List[Dict[str, Any]],
    model: RewardModel,
    mode: str = "absolute",
    reward_range: tuple = (-1, 1),
    device: Union[str, torch.device] = "cpu",
) -> List[Dict[str, Any]]:
    """
    以小批次對 replay 逐批推論 RewardModel，並將 reward 重新標注。
    支援模式：
      - "absolute"/"abs"：r = r(next_image)
      - "delta"/"diff"  ：r = r(next_image) - r(image)
    之後進行 z-score 標準化與 clip。
    """
    # --- 基本設定 ---
    dev = device if isinstance(device, torch.device) else torch.device(device)
    use_amp = dev.type == "cuda"
    mode_lower = mode.lower()
    if mode_lower not in ("absolute", "abs", "delta", "diff"):
        raise ValueError("mode must be 'absolute'/'abs' or 'delta'/'diff'")

    BATCH_SIZE = 32  # 小批次推論，避免一次吃爆顯存

    # --- 逐批推論，僅保留 CPU 側的 raw rewards ---
    raw_rewards: List[float] = []
    model.eval()

    with torch.no_grad():
        for start in range(0, len(replay), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(replay))
            batch = replay[start:end]

            # 準備本批影像（已在 preprocess_image 轉成 (C,H,W) tensor）
            imgs = [preprocess_image(t["image"]) for t in batch]
            nxts = [preprocess_image(t["next_image"]) for t in batch]

            x = torch.stack(imgs, dim=0).to(dev)
            x_next = torch.stack(nxts, dim=0).to(dev)

            with autocast(enabled=use_amp):
                r_curr = model(x)
                r_next = model(x_next)

            if mode_lower in ("absolute", "abs"):
                raw = r_next.detach().cpu().numpy()
            else:  # "delta"/"diff"
                raw = (r_next - r_curr).detach().cpu().numpy()

            raw_rewards.extend(raw.tolist())

            # 釋放本批 GPU 張量（交由 Python GC）
            del x, x_next, r_curr, r_next

    # --- 標準化與裁剪 ---
    raw_arr = np.asarray(raw_rewards, dtype=np.float32)
    mean = float(np.mean(raw_arr))
    std = float(np.std(raw_arr) + 1e-6)
    rewards = (raw_arr - mean) / std

    low, high = reward_range
    rewards = np.clip(rewards, low, high)

    # --- 回寫到 replay ---
    for t, r in zip(replay, rewards):
        t["reward"] = float(r)

    return replay
