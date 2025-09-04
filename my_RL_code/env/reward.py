import numpy as np
import cv2
from vlm import get_alignment_score

def reward_horizontal_alignment(image: np.ndarray, threshold: int = 10) -> float:
    """
    保留原本的傳統 CV 水平排列評分函式（向後相容）。
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) < 2:
        return 0.0

    centers_y = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h < 300:
            continue
        cy = y + h // 2
        centers_y.append(cy)

    if len(centers_y) < 2:
        return 0.0

    variance = np.var(centers_y)
    score = max(0.0, 1.0 - (variance / (threshold ** 2)))
    return float(score)

def vlm_reward(image: np.ndarray, instruction: str = "align objects in a row") -> float:
    """
    使用 VLM 模型語意評分。
    """
    return get_alignment_score(image, instruction=instruction)

def reward_horizontal_alignment_combined(image: np.ndarray, threshold: int = 10) -> float:
    """
    混合 reward 策略：CV + VLM。
    """
    cv_score = reward_horizontal_alignment(image, threshold)
    vlm_score = vlm_reward(image)
    return 0.5 * cv_score + 0.5 * vlm_score


# 更多 reward 函數可以加入這個檔案，例如：
# - reward_clustered_together(image)
# - reward_all_in_line(image)
# - reward_based_on_vlm(image)