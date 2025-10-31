import os
import json
import random
import shutil
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple
import torch


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
        self.images: List[np.ndarray] = []

    def add(self, image: np.ndarray) -> None:
        self.images.append(image)

    def sample_pairs(self, count: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        pairs = []
        n = len(self.images)
        for _ in range(count):
            i, j = random.sample(range(n), 2)
            pairs.append((self.images[i], self.images[j]))
        return pairs

    def __len__(self):
        return len(self.images)

    def clear(self):
        self.images.clear()


# ========== Preference Dataset ==========
class PreferenceDataset:
    def __init__(self):
        self.data: List[Tuple[np.ndarray, np.ndarray, int]] = []

    def add(self, img0: np.ndarray, img1: np.ndarray, label: int):
        assert label in {0, 1}
        self.data.append((img0, img1, label))

    def __len__(self):
        return len(self.data)

    def clear(self):
        self.data.clear()

    def get_all(self) -> List[Tuple[np.ndarray, np.ndarray, int]]:
        return self.data


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
