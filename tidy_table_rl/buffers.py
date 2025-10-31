#!/usr/bin/env python3
"""
buffers.py

定義 RL-VLM-F 所需的三個緩衝區：
1. ImageBuffer (I): 儲存高品質影像，供 VLM 抽樣。
2. PreferenceDataset (D): 儲存 VLM 標註的偏好對。
3. ReplayBuffer (B): 儲存 (s, a, s', imgs...)，並提供 relabel 功能。
"""

import numpy as np
import random
from collections import deque
from typing import Tuple, List, Dict, Any, Callable

# 確保影像儲存為 uint8 以節省記憶體
Image = np.ndarray  # Type hint for HWC, uint8 image

# =====================================
# 📸 影像池 (I)
# =====================================

class ImageBuffer:
    """ 影像緩衝區 (I)，儲存高品質影像供 VLM 採樣 """
    def __init__(self, max_size: int):
        self.max_size = max_size
        self._buffer = deque(maxlen=max_size)

    def __len__(self) -> int:
        return len(self._buffer)

    def add(self, image: Image):
        """ 新增一張高品質影像 (uint8) """
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        self._buffer.append(image)

    def sample_pair(self) -> Tuple[Image, Image] | None:
        """ 隨機抽取一對影像 (sigma_0, sigma_1) """
        if len(self._buffer) < 2:
            return None
        
        # 確保抽到兩張不同的影像
        idx1, idx2 = random.sample(range(len(self._buffer)), 2)
        return self._buffer[idx1], self._buffer[idx2]

# =====================================
# 👍 偏好資料集 (D)
# =====================================

@np.record
class Preference:
    """ 儲存一個偏好樣本 """
    image_0: Image
    image_1: Image
    label: int  # 0 (偏好 0), 1 (偏好 1)

class PreferenceDataset:
    """ 偏好資料集 (D)，儲存 VLM 的標註結果 """
    def __init__(self, max_size: int):
        self.max_size = max_size
        self._buffer = deque(maxlen=max_size)

    def __len__(self) -> int:
        return len(self._buffer)

    def add(self, image_0: Image, image_1: Image, label: int):
        """ 新增一筆偏好 (y=0 或 y=1) """
        if label not in (0, 1):
            raise ValueError("標籤 'label' 必須是 0 或 1")
        
        # 確保 uint8
        if image_0.dtype != np.uint8:
            image_0 = np.clip(image_0, 0, 255).astype(np.uint8)
        if image_1.dtype != np.uint8:
            image_1 = np.clip(image_1, 0, 255).astype(np.uint8)
            
        self._buffer.append(Preference(image_0, image_1, label))

    def sample(self, batch_size: int) -> List[Preference]:
        """ 隨機抽樣一個 batch 的偏好 """
        batch_size = min(batch_size, len(self._buffer))
        return random.sample(self._buffer, batch_size)

# =====================================
# 🔄 重播緩衝區 (B)
# =====================================

class ReplayBuffer:
    """ 
    重播緩衝區 (B)，儲存 (s, a, s', img_t, img_{t+1}, done)。
    核心功能: relabel()，使用 r_psi 計算差分獎勵。
    """
    def __init__(self, state_dim: int, action_dim: int, max_size: int):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        # 狀態 (s, s')
        self.state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.next_state = np.zeros((max_size, state_dim), dtype=np.float32)
        # 動作 (a)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        # 影像 (img_t, img_{t+1}) - 使用 object array 儲存
        self.image = np.empty((max_size, 1), dtype=object)
        self.next_image = np.empty((max_size, 1), dtype=object)
        # 獎勵 (r) - 等待 relabel
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        # 結束 (done)
        self.done = np.zeros((max_size, 1), dtype=np.float32)

    def __len__(self) -> int:
        return self.size

    def add(self, 
            state: np.ndarray, 
            action: np.ndarray, 
            next_state: np.ndarray, 
            image: Image,         # image_t
            next_image: Image,    # image_{t+1}
            done: bool):
        """
        新增一筆 transition。
        注意: reward 預設為 0，等待 relabel()。
        """
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.image[self.ptr, 0] = image
        self.next_image[self.ptr, 0] = next_image
        self.reward[self.ptr] = 0.0  # 預設為 0
        self.done[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def relabel(self, reward_model_fn: Callable[[Image], float]):
        """
        使用新的 r_psi 模型重新標註**所有**獎勵。
        r_t = r_psi(img_{t+1}) - r_psi(img_t)
        """
        if self.size == 0:
            return

        print(f"[Relabel] 正在為 {self.size} 筆 transition 重新計算獎勵...")
        
        # 1. 批次計算所有 img_t 和 img_{t+1} 的分數
        # (這裡假設 reward_model_fn 可以接受批次輸入，如果不行，需改為 for 迴圈)
        try:
            # 取出所有影像
            imgs_t = np.stack(self.image[:self.size, 0])      # (size, H, W, 3)
            imgs_t_plus_1 = np.stack(self.next_image[:self.size, 0]) # (size, H, W, 3)

            # 批次打分
            scores_t = reward_model_fn(imgs_t)          # (size,)
            scores_t_plus_1 = reward_model_fn(imgs_t_plus_1)  # (size,)

        except ValueError: 
            # 如果堆疊失敗 (例如影像大小不一) 或模型不支援批次，退回 for 迴圈
            print("[Relabel] 批次處理失敗，退回 for 迴圈 (較慢)")
            scores_t = np.array([reward_model_fn(img) for img in self.image[:self.size, 0]])
            scores_t_plus_1 = np.array([reward_model_fn(img) for img in self.next_image[:self.size, 0]])
        
        # 2. 計算差分獎勵
        diff_rewards = scores_t_plus_1 - scores_t
        
        # 3. 覆寫緩衝區中的獎勵
        self.reward[:self.size] = diff_rewards.reshape(-1, 1)
        
        print(f"[Relabel] 完成。獎勵平均值: {np.mean(diff_rewards):.4f}")

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """ 
        為 SAC (Policy) 抽樣一個 batch 的 (s, a, r, s', done)。
        注意: 這裡回傳的 r 是已經被 relabel 過的。
        """
        idxs = np.random.randint(0, self.size, size=batch_size)

        return {
            "state": self.state[idxs],
            "action": self.action[idxs],
            "reward": self.reward[idxs],  # 回傳已 relabel 的獎勵
            "next_state": self.next_state[idxs],
            "done": self.done[idxs]
            # 注意: SAC 不需要影像，所以這裡不回傳
        }