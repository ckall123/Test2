import threading
import queue
from functools import lru_cache
import hashlib
import numpy as np

# 這裡仍用 stub（0.5），你之後可換成真正的 VLM API 呼叫（e.g. HTTP/本地模型）

def _hash_image(img: np.ndarray) -> str:
    return hashlib.md5(img.tobytes()).hexdigest() if img is not None else "none"

@lru_cache(maxsize=4096)
def _score_stub(img_hash: str) -> float:
    return 0.5

class AsyncVLMClient:
    def __init__(self, batch_size=4):
        self.q = queue.Queue(maxsize=64)
        self.latest = 0.0
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        self.batch_size = batch_size

    def submit(self, image_bgr):
        try:
            self.q.put_nowait(image_bgr)
        except queue.Full:
            pass

    def get_latest(self, default=0.0) -> float:
        return float(self.latest) if self.latest is not None else float(default)

    def _worker(self):
        while True:
            batch = [self.q.get()]
            try:
                for _ in range(self.batch_size - 1):
                    batch.append(self.q.get_nowait())
            except queue.Empty:
                pass
            # 批量評分（此處 stub 用 hash 快取）
            scores = []
            for img in batch:
                try:
                    h = _hash_image(img)
                    s = _score_stub(h)
                except Exception:
                    s = 0.0
                scores.append(s)
            # 取最後一張當作最新（你也可以平均/中位數）
            if len(scores) > 0:
                self.latest = float(scores[-1])