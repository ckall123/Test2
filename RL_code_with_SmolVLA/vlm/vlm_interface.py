def get_vlm_score(image_bgr, prompt: str = "請評分桌面整潔度，0~1") -> float:
    if image_bgr is None:
        return 0.0
    # TODO: 換成你真的 VLM / 本地 reward model
    return 0.5
