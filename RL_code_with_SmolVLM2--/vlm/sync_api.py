"""
Synchronous VLM client (Ollama)
- Minimal deps (urllib + Pillow). No streaming.
- Works with vision models like qwen2.5-vl / qwen3-vl.

Usage:
    from vlm.sync_api import score_image
    s = score_image(np_image, prompt="Rate tidiness 0..1 only")

Env vars (optional):
    OLLAMA_HOST  : default "http://localhost:11434"
    OLLAMA_VLM   : default "qwen2.5-vl"
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import urllib.request
from typing import Optional

import numpy as np

try:
    from PIL import Image
    _PIL_OK = True
except Exception:  # pragma: no cover
    _PIL_OK = False


def _np_to_png_b64(img: np.ndarray) -> Optional[str]:
    if not _PIL_OK:
        return None
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    if img.ndim == 2:
        mode = "L"
    elif img.ndim == 3 and img.shape[2] == 3:
        mode = "RGB"
    elif img.ndim == 3 and img.shape[2] == 4:
        mode = "RGBA"
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")
    im = Image.fromarray(img, mode=mode)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    b = base64.b64encode(buf.getvalue()).decode("ascii")
    return b


def _post_json(url: str, payload: dict, timeout: float = 30.0) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="ignore")
    try:
        return json.loads(raw)
    except Exception:
        return {"response": raw}


def _extract_float(text: str, lo: float = 0.0, hi: float = 1.0) -> Optional[float]:
    # find first float-like token in text
    m = re.search(r"-?\d+(?:\.\d+)?", text)
    if not m:
        return None
    try:
        v = float(m.group(0))
    except Exception:
        return None
    return float(max(lo, min(hi, v)))


def score_image(
    image_np: np.ndarray,
    *,
    prompt: str = (
        "You are a tidy desk judge. Score how tidy this tabletop arrangement looks. "
        "Return a single number from 0 to 1 where 1=very tidy, 0=very messy. Output only the number."
    ),
    host: Optional[str] = None,
    model: Optional[str] = None,
    timeout: float = 45.0,
) -> Optional[float]:
    """Return a scalar score in [0,1] from a VLM via Ollama. None on failure."""
    host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
    model = model or os.getenv("OLLAMA_VLM", "qwen2.5-vl")

    b64 = _np_to_png_b64(image_np)
    if b64 is None:
        return None

    payload = {
        "model": model,
        "prompt": prompt,
        "images": [b64],  # vision models accept images[]
        "stream": False,
        # temperature left default for determinism
    }
    try:
        res = _post_json(host.rstrip("/") + "/api/generate", payload, timeout=timeout)
    except Exception:
        return None

    text = (res.get("response") or "").strip()
    val = _extract_float(text, 0.0, 1.0)
    return val
