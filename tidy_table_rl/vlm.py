import base64
import json
from io import BytesIO
import numpy as np
from PIL import Image
from ollama import chat


class VLMScorer:
    def __init__(self, model: str = "llama3.2-vision:11b"):
        self.model = model

    def _encode_image(self, image: np.ndarray) -> str:
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        buf = BytesIO()
        Image.fromarray(image).save(buf, format="JPEG")
        return base64.b64encode(buf.getvalue()).decode()

    def _extract_json_score(self, text: str) -> float:
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            l, r = text.find("{"), text.rfind("}")
            obj = json.loads(text[l:r+1]) if l != -1 and r != -1 and r > l else {"score": 0.0}
        return max(0.0, min(float(obj.get("score", 0.0)), 1.0))

    def score_image(self, image: np.ndarray, instruction: str = "align objects in a row") -> float:
        encoded = self._encode_image(image)
        prompt = (
            "You are a strict visual judge for desk tidiness.\n"
            "Return ONLY JSON: {\"score\": <float in [0,1]>}.\n"
            f"Instruction: {instruction}"
        )
        try:
            resp = chat(model=self.model, messages=[{
                "role": "user",
                "content": prompt,
                "images": [f"data:image/jpeg;base64,{encoded}"]
            }])
            return self._extract_json_score(resp['message']['content'])
        except Exception:
            return 0.0

    def compare(self, img_a: np.ndarray, img_b: np.ndarray, rubric: str) -> int:
        enc_a = self._encode_image(img_a)
        enc_b = self._encode_image(img_b)

        analysis = chat(model=self.model, messages=[{
            "role": "user",
            "content": (
                f"You are evaluating two desk layouts (Image A and B) for neatness.\n"
                f"The judging rubric is:\n{rubric}\n\n"
                f"Image A:\n<img src='data:image/jpeg;base64,{enc_a}'>\n\n"
                f"Image B:\n<img src='data:image/jpeg;base64,{enc_b}'>\n\n"
                "Your analysis:"
            )
        }])['message']['content']

        try:
            decision = chat(model=self.model, messages=[
                {"role": "user", "content": analysis},
                {"role": "user", "content": (
                    "Based on the previous analysis, choose the better image.\n"
                    "Return ONE number: 0 if A is better, 1 if B is better, -1 if indistinguishable."
                )}
            ])['message']['content'].strip()

            return 0 if "0" in decision else 1 if "1" in decision else -1
        except Exception:
            return -1
        
        
if __name__ == '__main__':
    import cv2

    img_a = cv2.imread("topdown_view.jpg")
    img_b = cv2.imread("topdown_view2.jpg")
    img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB)
    img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB)

    rubric = (
        "- Items are aligned in straight rows or columns.\n"
        "- Empty space between items is balanced.\n"
        "- No excessive overlap or clutter.\n"
        "- Object orientations are consistent."
    )

    scorer = VLMScorer()
    label = scorer.compare(img_a, img_b, rubric)
    print(f"Preference: {label}")
