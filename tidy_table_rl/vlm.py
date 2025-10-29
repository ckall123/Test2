#!/usr/bin/env python3
"""
VLMScorer 模組（改為本地 llama3.2-vision:11b）：
使用 Ollama 本地模型進行桌面整潔度評分，不透過 HTTP API。
"""

import base64
import json
from io import BytesIO
from typing import Optional

import numpy as np
from PIL import Image
from ollama import chat


class VLMScorer:
    def __init__(self, model: str = "llama3.2-vision:11b"):
        self.model = model

    def _encode_image(self, image: np.ndarray) -> str:
        """將 NumPy RGB 圖像轉為 base64 的 JPEG；供本地 VLM multimodal 輸入。"""
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image 必須為 HxWx3 的 RGB NumPy 陣列")
        buffer = BytesIO()
        Image.fromarray(image).save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _extract_json_score(self, text: str) -> float:
        """解析模型輸出中的 JSON，抓取 {"score": float}，夾在 [0,1] 範圍內。"""
        try:
            obj = json.loads(text)
            score = float(obj.get("score", 0.0))
            return max(0.0, min(score, 1.0))
        except Exception:
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
        """使用本地 llama3.2-vision 模型給圖片打分（回傳 0~1 整齊度）。"""
        encoded_image = self._encode_image(image)
        data_url = f"data:image/jpeg;base64,{encoded_image}"

        prompt = (
            "You are a strict visual judge for desk tidiness.\n"
            "Return ONLY JSON: {\"score\": <float in [0,1]>}.\n"
            f"Instruction: {instruction}"
        )

        try:
            response = chat(model=self.model, messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [data_url]
                }
            ])
            result = response['message']['content']
            return self._extract_json_score(result)
        except Exception as e:
            print(f"[VLM ERROR] {e}")
            return 0.0


if __name__ == '__main__':
    # 單檔測試
    import cv2

    def main():
        image_path = "A.jpg"
        print(image_path)
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            print(f"[ERROR] 圖片讀取失敗: {image_path}")
            return
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        scorer = VLMScorer(model="llama3.2-vision:11b")
        score = scorer.score_image(image_rgb, instruction="align objects in a row")
        print(f"[VLM score] {score:.3f}")

    main()
