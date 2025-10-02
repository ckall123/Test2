import numpy as np
import cv2
from tidy_table_rl.vlm import VLMScorer


def reward_alignment(image: np.ndarray, min_area: int = 300, y_var_norm: float = 10.0) -> float:
    """
    幾何對齊獎勵：
    根據物體輪廓的 Y 軸中心點變異數評估整齊程度。
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centers_y = [y + h * 0.5 for x, y, w, h in (cv2.boundingRect(c) for c in contours) if w * h >= min_area]
    if len(centers_y) < 2:
        return 0.0

    y_var = float(np.var(centers_y))
    return max(0.0, 1.0 - (y_var / (y_var_norm ** 2)))


def reward_vlm(image: np.ndarray, instruction: str = "align objects in a row") -> float:
    """
    視覺語言模型獎勵：
    由 VLM 模型對整潔程度給分（0~1）。
    """
    scorer = VLMScorer()
    return scorer.score_image(image, instruction)


def reward_combined(image: np.ndarray, instruction: str = "align objects in a row") -> float:
    """
    混合獎勵：結合幾何與語意兩種方法。
    """
    r_align = reward_alignment(image)
    r_vlm = reward_vlm(image, instruction)
    return 0.7 * r_align + 0.3 * r_vlm


# 預設獎勵函式（推薦使用語意型 VLM 評分）
REWARD_FN = reward_combined



# 更多 reward 函數可以加入這個檔案，例如：
# - reward_clustered_together(image)
# - reward_all_in_line(image)
# - reward_based_on_vlm(image)