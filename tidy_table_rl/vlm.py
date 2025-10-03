#!/usr/bin/env python3
"""
VLMScorer 模組：
結合視覺語言模型（例如 Qwen2/3 via Ollama）給圖片打分，用於整潔度、美觀度等任務的獎勵訊號。
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
    使用視覺語言模型 (VLM) 針對圖像進行任務導向的評分。
    傳入圖像與描述指令，取得 0~1 的分數作為整潔度等任務的指標。
    """
    def __init__(self, endpoint: str = "http://localhost:11434/api/generate", model: str = "qwen2.5vl:latest"):
        self.endpoint = endpoint
        self.model = model

    def _encode_image(self, image: np.ndarray) -> str:
        """將 NumPy 圖像轉為 base64 編碼的 JPEG 格式。"""
        buffer = BytesIO()
        Image.fromarray(image).save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def score_image(self, image: np.ndarray, instruction: str = "align objects in a row") -> float:
        """
        傳送圖片與任務指令給模型，取得 0~1 的分數。

        :param image: 輸入圖像（NumPy 格式，RGB）
        :param instruction: 任務指令（例如整齊擺放）
        :return: 分數（0~1），若失敗則為 0.0
        """
        encoded_image = self._encode_image(image)
        prompt = (
            "You are a strict visual judge. Return ONLY JSON: {\"score\": <float in [0,1]>}.\n"
            f"Instruction: {instruction}"
        )

        payload = {
            "model": self.model,
            "prompt": prompt,
            "images": [encoded_image],
            "stream": False
        }

        try:
            response = requests.post(self.endpoint, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json().get("response", "")

            json_part = result[result.find("{"):result.rfind("}")+1]
            score = float(json.loads(json_part).get("score", 0.0))
            return max(0.0, min(score, 1.0))

        except Exception as e:
            print(f"[VLM ERROR] {e}")
            return 0.0


if __name__ == '__main__':
    import cv2
    import sys

    def main():
        image_path = sys.argv[1] if len(sys.argv) > 1 else "my_RL_code/A.jpg"
        image = cv2.imread(image_path)

        if image is None:
            print(f"[ERROR] 圖片讀取失敗: {image_path}")
            return

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        scorer = VLMScorer()
        score = scorer.score_image(image_rgb)
        print(f"[VLM score] {score:.3f}")

    main()
