import random
from typing import List, Tuple
from dataclasses import dataclass
import numpy as np

@dataclass
class Transition:
    obs: np.ndarray
    action: np.ndarray
    reward: float
    next_obs: np.ndarray
    done: bool
    info: dict

class ReplayBuffer:
    """
    儲存 RL agent 的 transition 資料。
    支援 sample 與 reward 更新（relabeling）。
    """
    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        self.buffer: List[Transition] = []

    def add(self, obs, action, reward, next_obs, done, info={}):
        """新增一筆 transition"""
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(Transition(obs, action, reward, next_obs, done, info))

    def sample_batch(self, batch_size: int) -> List[Transition]:
        """隨機取出一批 transition"""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def update_rewards(self, new_rewards: List[float]):
        """
        根據給定的新 reward 值，更新 buffer 中對應筆數。
        通常搭配 reward model 重算 reward 使用。
        """
        for i, r in enumerate(new_rewards):
            if i < len(self.buffer):
                self.buffer[i].reward = r

    def __len__(self):
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()