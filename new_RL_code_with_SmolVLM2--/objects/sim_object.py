import numpy as np

class SimObject:
    """
    場景物體資料模型：名稱、位置、連結、是否被抓取。
    """
    def __init__(self, name: str, position: np.ndarray, link: str = "body"):
        self.name = name
        self.position = position.astype(np.float32)
        self.link = link
        self.attached = False

    def is_near(self, point: np.ndarray, thresh: float = 0.08) -> bool:
        return float(np.linalg.norm(self.position - point)) < float(thresh)

    def __repr__(self):
        return f"SimObject(name={self.name}, position={self.position}, attached={self.attached})"