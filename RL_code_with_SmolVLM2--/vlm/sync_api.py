# vlm/sync_api.py
# 使用 Ollama 的 Qwen 2.5-VL 做整潔度評分（0~1），無外部依賴，只需本機跑著 ollama。
#   拉模型：  ollama pull qwen2.5vl
#   端點：    http://localhost:11434/api/chat

from __future__ import annotations
import base64, json, re
from typing import Optional

import cv2
import numpy as np
import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "qwen2.5vl"

JUDGE_PROMPT = (
    "You are a strict desk-tidiness judge.\n"
    "Score how tidy this tabletop arrangement is considering: alignment to a straight line, "
    "even spacing, no overlaps, and objects placed within the table area.\n"
    'Return ONLY JSON: {"score": float in [0,1], "reason": "one sentence"}'
)


def _img_to_b64(img_bgr: np.ndarray) -> str:
    if img_bgr is None:
        raise ValueError("img_bgr is None")
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return base64.b64encode(buf.tobytes()).decode()


def get_vlm_score(img_bgr: np.ndarray,
                  model: str = DEFAULT_MODEL,
                  timeout_s: int = 45) -> float:
    """回傳 0~1 的整潔度分數。若呼叫失敗則回 0.0。"""
    b64 = _img_to_b64(img_bgr)
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": JUDGE_PROMPT,
            "images": [b64],
        }],
        "options": {"temperature": 0},
        # 讓模型以 JSON schema 格式化輸出（Ollama 支援 format）
        "format": {
            "type": "object",
            "properties": {
                "score": {"type": "number"},
                "reason": {"type": "string"},
            },
            "required": ["score", "reason"],
        },
    }

    content: Optional[str] = None
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=timeout_s)
        r.raise_for_status()
        content = r.json().get("message", {}).get("content", "{}")
        data = json.loads(content)
        s = float(data.get("score", 0.0))
        return max(0.0, min(1.0, s))
    except Exception:
        # 後備：嘗試從回傳文字抓第一個小數
        if content:
            m = re.search(r"([0]?\.?\d+)", content)
            if m:
                try:
                    s = float(m.group(1))
                    return max(0.0, min(1.0, s))
                except Exception:
                    pass
        return 0.0


# （可選）雙圖偏好，用於收集少量偏好資料或做評比
PAIR_PROMPT = (
    "You will compare two images A and B of a tabletop.\n"
    "Pick which is tidier based on alignment, even spacing, no overlaps, and objects within table area.\n"
    'Return ONLY JSON: {"winner":"A"|"B", "reason":"one sentence"}'
)


def prefer_A_over_B(imgA_bgr: np.ndarray,
                    imgB_bgr: np.ndarray,
                    model: str = DEFAULT_MODEL,
                    timeout_s: int = 60) -> int:
    b64A, b64B = _img_to_b64(imgA_bgr), _img_to_b64(imgB_bgr)
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": PAIR_PROMPT,
            "images": [b64A, b64B],
        }],
        "options": {"temperature": 0},
        "format": {
            "type": "object",
            "properties": {"winner": {"type": "string"}, "reason": {"type": "string"}},
            "required": ["winner", "reason"],
        },
    }
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=timeout_s)
        r.raise_for_status()
        data = json.loads(r.json().get("message", {}).get("content", "{}"))
        return +1 if str(data.get("winner", "A")).upper() == "A" else -1
    except Exception:
        return 0