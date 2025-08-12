import random
from typing import List, Tuple
from dataclasses import dataclass

@dataclass
class ImageEntry:
    episode_id: int
    step_index: int
    image: any  # placeholder for actual image (e.g. np.ndarray or ROS msg)
    metadata: dict

class ImageBuffer:
    """
    儲存來自環境相機的畫面，用於後續偏好學習（preference learning）。
    可在任意時間點抽取兩張圖片做 pairwise 比較。
    """
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer: List[ImageEntry] = []

    def add(self, episode_id: int, step_index: int, image: any, metadata: dict = {}):
        """新增一筆圖像與該時間點相關資訊"""
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(ImageEntry(episode_id, step_index, image, metadata))

    def sample_pairs(self, num_pairs: int) -> List[Tuple[ImageEntry, ImageEntry]]:
        """
        隨機取出若干組圖片 pair，用於送入 VLM 做偏好比較
        回傳 [(imgA, imgB), ...]
        """
        pairs = []
        if len(self.buffer) < 2:
            return pairs

        for _ in range(num_pairs):
            a, b = random.sample(self.buffer, 2)
            pairs.append((a, b))
        return pairs

    def __len__(self):
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()