#!/usr/bin/env python3
"""
VLMScorer 模組：
在你的「整理桌面」實驗中，透過本模組呼叫本地 Ollama (e.g., Qwen2/3-VL) 對相機截圖打分，
輸出 0~1 分數，供環境內的「受控 VLM 加分」shaping 使用（僅作輔助，不決定 done）。
"""

import base64
import json
from io import BytesIO
from typing import Optional

import numpy as np
import requests
from PIL import Image


class VLMScorer:
    """
    使用視覺語言模型 (VLM) 針對圖像進行任務導向評分。
    - 輸入：RGB 圖像 (NumPy array) + 任務指令（如「align objects in a row」）
    - 輸出：0~1 的整潔度分數（用於 ΔE>threshold 時的輔助加分）
    """
    def __init__(self, endpoint: str = "http://localhost:11434/api/generate", model: str = "qwen2.5vl:latest", timeout: float = 30.0):
        self.endpoint = endpoint
        self.model = model
        self.timeout = timeout

    def _encode_image(self, image: np.ndarray) -> str:
        """將 NumPy RGB 圖像轉為 base64 的 JPEG；供 Ollama VLM multimodal 輸入。"""
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image 必須為 HxWx3 的 RGB NumPy 陣列")
        buffer = BytesIO()
        Image.fromarray(image).save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _extract_json_score(self, text: str) -> float:
        """
        解析模型輸出中的 JSON，抓取 {"score": float}，再夾在 [0,1]。
        實驗中避免因模型多話導致解析失敗。
        """
        try:
            # 嘗試整段直接當 JSON
            obj = json.loads(text)
            score = float(obj.get("score", 0.0))
            return max(0.0, min(score, 1.0))
        except Exception:
            # 從字串中擷取第一段 {...} 當 JSON
            l = text.find("{")
            r = text.rfind("}")
            if l != -1 and r != -1 and r > l:
                try:
                    obj = json.loads(text[l:r+1])
                    score = float(obj.get("score", 0.0))
                    return max(0.0, min(score, 1.0))
                except Exception:
                    pass
        return 0.0

    def score_image(self, image: np.ndarray, instruction: str = "align objects in a row") -> float:
        """
        呼叫 Ollama VLM，回傳 0~1 分數。
        在你的環境中，該分數將以「步距節流 + ΔE 門檻」的規則作為 shaping 加分來源。
        """
        encoded_image = self._encode_image(image)
        prompt = (
            "You are a strict visual judge for desk tidiness.\n"
            "Return ONLY JSON: {\"score\": <float in [0,1]>}.\n"
            f"Instruction: {instruction}"
        )

        payload = {
            "model": self.model,
            "prompt": prompt,
            "images": [encoded_image],
            "stream": False
        }

        try:
            resp = requests.post(self.endpoint, json=payload, timeout=self.timeout, headers={"Content-Type": "application/json"})
            resp.raise_for_status()
            # Ollama /api/generate 回傳欄位為 "response"
            raw = resp.json().get("response", "")
            return self._extract_json_score(raw)
        except Exception as e:
            print(f"[VLM ERROR] {e}")
            return 0.0


if __name__ == '__main__':
    # 單檔測試：用於你在整合前驗證「VLM 打分」是否可用
    import cv2

    def main():
        image_path = "topdown_view.jpg"
        print(image_path)
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            print(f"[ERROR] 圖片讀取失敗: {image_path}")
            return
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        scorer = VLMScorer()
        score = scorer.score_image(image_rgb, instruction="align objects in a row")
        print(f"[VLM score] {score:.3f}")

    main()
