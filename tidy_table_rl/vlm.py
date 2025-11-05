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
        """
        (偏好教師) Two-Stage Prompting: 比較兩張圖的整齊度，回傳偏好標籤。
        0=偏好 A (第一張圖)，1=偏好 B (第二張圖)，-1=無法分辨。
        """
        try:
            enc_a = self._encode_image(img_a)
            enc_b = self._encode_image(img_b)
            
            # ⭐ 修正: 重新使用 HTML 嵌入作為穩定的多圖傳輸方式 ⭐
            image_html_a = f'<img src="data:image/jpeg;base64,{enc_a}">'
            image_html_b = f'<img src="data:image/jpeg;base64,{enc_b}">'

            # --- Stage 1: Analysis ---
            analysis_prompt = (
                f"You are evaluating two desk layouts. Image A is:\n{image_html_a}\n\n"
                f"Image B is:\n{image_html_b}\n\n"
                f"The judging rubric is:\n{rubric}\n\n"
                "Analyze and describe how well each image (A and B) aligns with the rubric. Be detailed."
            )

            # 呼叫 chat 時不傳入 'images' 鍵，因為圖片已嵌入 content
            analysis_response = chat(model=self.model, messages=[{
                "role": "user", 
                "content": analysis_prompt
            }])
            analysis = analysis_response['message']['content']

            # --- Stage 2: Decision ---
            decision_prompt = (
                "Based ONLY on the previous analysis, choose the better image according to the rubric.\n"
                "Return ONE number: 0 if A is better, 1 if B is better, -1 if indistinguishable."
            )
            
            decision_response = chat(model=self.model, messages=[
                {"role": "user", "content": analysis},
                {"role": "user", "content": decision_prompt}
            ])
            decision = decision_response['message']['content'].strip()

            # --- Final Label Output ---
            return 0 if "0" in decision else 1 if "1" in decision else -1
            
        except Exception as e:
            # API 呼叫失敗，回傳 -1，避免程式崩潰
            # print(f"[VLMScorer ERROR] compare failed: {e}") 
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
