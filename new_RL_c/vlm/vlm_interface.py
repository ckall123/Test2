from functools import lru_cache
import hashlib
import numpy as np

def _hash_image(img: np.ndarray) -> str:
    return hashlib.md5(img.tobytes()).hexdigest() if img is not None else "none"

@lru_cache(maxsize=2048)
def _cached_score(img_hash: str, prompt: str) -> float:
    # TODO: 接真 VLM 後替換；先忽略 prompt，但保留介面
    return 0.5

def get_vlm_score(image_bgr, prompt: str = "請評分桌面整潔度，0~1") -> float:
    if image_bgr is None: return 0.0
    try:
        img_hash = _hash_image(image_bgr)
        s = float(_cached_score(img_hash, prompt))
        return max(0.0, min(1.0, s))
    except Exception:
        return 0.0