import cv2
import numpy as np


def to_bgr_uint8(img) -> np.ndarray:
    """保守轉換，確保為 BGR uint8。"""
    if img is None:
        return None
    if isinstance(img, np.ndarray):
        if img.dtype != np.uint8:
            x = img
            if x.max() <= 1.0:
                x = (x * 255.0).clip(0, 255)
            return x.astype(np.uint8)
        return img
    return np.asarray(img, dtype=np.uint8)


def resize_bgr(img: np.ndarray, w: int, h: int) -> np.ndarray:
    img = to_bgr_uint8(img)
    if img is None:
        return None
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)