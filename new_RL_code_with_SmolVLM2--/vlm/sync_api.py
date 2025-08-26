import numpy as np
from .vlm_core import _hash_image, _cached_score


def get_vlm_score(image_bgr: np.ndarray, prompt: str = "請評分桌面整潔度，0~1") -> float:
    if image_bgr is None:
        return 0.0
    try:
        img_hash = _hash_image(image_bgr)
        s = float(_cached_score(img_hash, prompt))
        return max(0.0, min(1.0, s))
    except Exception:
        return 0.0