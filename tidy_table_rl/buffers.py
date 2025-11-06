# buffers.py
# ------------------------------------------------------
# Contains ReplayBuffer, ImageBuffer, and PreferenceDataset
# Each buffer supports core functionalities for RL-VLM-F
# ------------------------------------------------------

import random
import numpy as np
from typing import List, Dict, Any, Tuple


# ========== Replay Buffer ==========
class ReplayBuffer:
    def __init__(self):
        self.data: List[Dict[str, Any]] = []

    def add(self, transition: Dict[str, Any]) -> None:
        self.data.append(transition)

    def sample(self, batch_size: int) -> List[Dict[str, Any]]:
        return random.sample(self.data, batch_size)

    def __len__(self):
        return len(self.data)

    def clear(self):
        self.data.clear()


# ========== Image Buffer ==========
class ImageBuffer:
    def __init__(self):
        self.episodes: Dict[int, List[Tuple[int, np.ndarray]]] = {}

    def add(self, episode_id: int, step_id: int, image: np.ndarray):
        if episode_id not in self.episodes:
            self.episodes[episode_id] = []
        self.episodes[episode_id].append((step_id, image))

    def sample_pairs(self, count: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        pairs = []
        for episode_id, frames in self.episodes.items():
            frames.sort()  # by step_id
            n = len(frames)
            for _ in range(min(count, n // 2)):
                if n < 2: continue
                i = random.randint(0, n - 2)
                j = min(n - 1, i + random.randint(1, 5))
                pairs.append((frames[i][1], frames[j][1]))
        # 加入少量強對比樣本
        flat_images = [img for frames in self.episodes.values() for _, img in frames]
        for _ in range(min(5, len(flat_images) // 2)):
            i, j = random.sample(range(len(flat_images)), 2)
            pairs.append((flat_images[i], flat_images[j]))
        return pairs

    def __len__(self):
        return sum(len(v) for v in self.episodes.values())

    def clear(self):
        self.episodes.clear()


# ========== Preference Dataset ==========
class PreferenceDataset:
    def __init__(self):
        self.data: List[Tuple[np.ndarray, np.ndarray, int]] = []
        self.reject_count = 0

    def add(self, img0: np.ndarray, img1: np.ndarray, label: int):
        if label not in {0, 1}:
            self.reject_count += 1
            return
        self.data.append((img0, img1, label))

    def __len__(self):
        return len(self.data)

    def clear(self):
        self.data.clear()
        self.reject_count = 0

    def get_all(self) -> List[Tuple[np.ndarray, np.ndarray, int]]:
        return self.data

    def get_reject_rate(self) -> float:
        total = len(self.data) + self.reject_count
        return self.reject_count / total if total > 0 else 0.0
