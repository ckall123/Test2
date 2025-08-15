import numpy as np

class SimObject:
    """
    表示場景中一個物體的資料模型。
    用於記錄位置、抓取狀態等資訊。
    """
    def __init__(self, name: str, position: np.ndarray):
        self.name = name                    # 物體名稱（對應模型名）
        self.position = position            # 物體座標位置
        self.attached = False               # 是否被抓取

    def __repr__(self):
        return f"SimObject(name={self.name}, position={self.position}, attached={self.attached})"
