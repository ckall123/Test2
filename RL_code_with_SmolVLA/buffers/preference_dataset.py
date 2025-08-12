from typing import List, Tuple
from dataclasses import dataclass

@dataclass
class PreferenceEntry:
    image_a: any  # 可以是 image array, path, 或 encoded version
    image_b: any
    preference: int  # 0: A 比較好, 1: B 比較好
    meta: dict

class PreferenceDataset:
    """
    收集 VLM 回傳的偏好資料，用於訓練 reward model。
    """
    def __init__(self):
        self.data: List[PreferenceEntry] = []

    def add(self, image_a, image_b, preference: int, meta: dict = {}):
        """新增一筆偏好資料：哪張圖片比較好 + 額外資訊"""
        self.data.append(PreferenceEntry(image_a, image_b, preference, meta))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> PreferenceEntry:
        return self.data[idx]

    def sample_batch(self, batch_size: int) -> List[PreferenceEntry]:
        import random
        return random.sample(self.data, min(batch_size, len(self.data)))

    def clear(self):
        self.data.clear()