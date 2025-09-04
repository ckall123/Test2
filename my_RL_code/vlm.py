import numpy as np
import base64
import requests
from PIL import Image
from io import BytesIO


class QwenVLMReward:
    def __init__(self, model: str = "qwen2:1.5b"):
        self.model = model

    def score(self, image: np.ndarray, instruction: str) -> float:
        pil_img = Image.fromarray(image)
        buffered = BytesIO()
        pil_img.save(buffered, format="JPEG")
        b64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}}
                ]}
            ]
        }

        try:
            response = requests.post("http://localhost:11434/api/chat", json=payload)
            response.raise_for_status()
            content = response.json()["message"]["content"].lower()
            if "yes" in content:
                return 1.0
            elif "no" in content:
                return 0.0
            else:
                return 0.5  # 不確定的情況
        except Exception as e:
            print(f"VLM scoring failed: {e}")
            return 0.0


# ✅ 提供統一接口
vlm_model = QwenVLMReward()

def get_alignment_score(image: np.ndarray, instruction: str = "align objects in a row") -> float:
    return vlm_model.score(image, instruction)
