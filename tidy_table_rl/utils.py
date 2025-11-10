import os
import json
import shutil
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple
import torch
import time
import rclpy

# ========== 通用工具 ==========
def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)

def save_json(obj: Dict, path: str | Path) -> None:
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)

def load_json(path: str | Path) -> Dict:
    with open(path, 'r') as f:
        return json.load(f)

def now_str() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")



# ========== Logger ==========
class CycleLogger:
    def __init__(self, save_dir: str):
        self.root = Path(save_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.metrics = []

    def log(self, info: Dict[str, Any]) -> None:
        self.metrics.append(info)

    def save(self):
        save_json(self.metrics, self.root / "log.json")

    def save_checkpoint(self, model, name="reward_model.pt"):
        torch.save(model.state_dict(), self.root / name)

    def save_image(self, image: np.ndarray, fname: str):
        from PIL import Image
        Image.fromarray(image).save(self.root / fname)

# ========== 一些簡化工具函式 ==========
def save_image_list(images: List[np.ndarray], folder: Path):
    ensure_dir(folder)
    for i, img in enumerate(images):
        path = folder / f"img_{i}.jpg"
        from PIL import Image
        Image.fromarray(img).save(path)

def backup_code(dst_folder: Path):
    code_files = [f for f in os.listdir('.') if f.endswith('.py')]
    for f in code_files:
        shutil.copy(f, dst_folder / f)
# -----------------------------------
def spin_until_ok(predicate, *, node=None, executor=None, period=0.05, timeout=10.0, min_stable=3):
    start = time.monotonic(); stable = 0
    while time.monotonic() - start < timeout:
        (executor.spin_once(timeout_sec=period) if executor else rclpy.spin_once(node, timeout_sec=period))
        if predicate():
            stable += 1
            if stable >= min_stable: return True
        else:
            stable = 0
    return False
# ========== Action History（給 env 組觀測） ==========
class ActionHistory:
    def __init__(self, max_len: int, action_dim: int):
        self.max_len = max_len
        self.action_dim = action_dim
        self.buffer: List[np.ndarray] = []

    def add(self, action: np.ndarray):
        self.buffer.append(np.asarray(action, dtype=np.float32))
        if len(self.buffer) > self.max_len:
            self.buffer.pop(0)

    def vector(self) -> np.ndarray:
        if not self.buffer:
            return np.zeros(self.max_len * self.action_dim, dtype=np.float32)
        arr = np.concatenate(self.buffer, axis=0)
        pad_len = self.max_len * self.action_dim - len(arr)
        if pad_len > 0:
            arr = np.concatenate([arr, np.zeros(pad_len, dtype=np.float32)], axis=0)
        return arr

    def clear(self):
        self.buffer.clear()

# ========== Image Resize（給 policy image） ==========
def preprocess_image(img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    from PIL import Image
    pil = Image.fromarray(img)
    resized = pil.resize(size, Image.BILINEAR)
    return np.array(resized)
