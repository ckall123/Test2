import torch as th
import torch.nn as nn
from PIL import Image
from transformers import AutoProcessor, AutoModel
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class SmolVLM2Encoder(nn.Module):
    """SmolVLM2 影像→特徵向量（正統：LazyLinear 決定 in_features）
    - 為什麼：SmolVLM2 的輸出維（CLS/POOL）不一定等於 config.hidden_size（常見 960/1024 差異）。
      用 LazyLinear 可在 *第一次前向* 自動綁定 in_features，同時參數於 __init__ 即註冊到 optimizer → 能學。
    - Freeze：只凍結 backbone；投影層與後續 MLP 可訓練。
    """
    def __init__(self, model_name="HuggingFaceTB/SmolVLM2-500M-Video-Instruct", out_dim=384, freeze=True):
        super().__init__()
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.backbone  = AutoModel.from_pretrained(model_name)
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.proj = nn.LazyLinear(out_dim)  # 參數於 __init__ 註冊，首輪前向自動決 in_features

    @th.no_grad()
    def encode(self, imgs_bchw: th.Tensor, prompt: str = ".") -> th.Tensor:
        # imgs_bchw: [B,3,H,W] float32 in [0,1]
        arr = (imgs_bchw.clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).cpu().numpy()  # [B,H,W,3]
        pil_images = [Image.fromarray(arr[i]) for i in range(arr.shape[0])]
        image_token = getattr(self.processor, "image_token", "<image>")
        base_text = prompt if (isinstance(prompt, str) and prompt.strip()) else "."
        texts = [f"{image_token} {base_text}"] * len(pil_images)
        inputs = self.processor(images=pil_images, text=texts, return_tensors="pt", padding=True)
        device = imgs_bchw.device
        inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        out = self.backbone(**inputs)
        last = out.last_hidden_state  # [B, L, D]
        feat = last[:, 0] if last.shape[1] > 0 else last.mean(dim=1)  # [B, D]
        return feat

    def forward(self, imgs_bchw: th.Tensor, prompt: str = ".") -> th.Tensor:
        with th.no_grad():
            feat = self.encode(imgs_bchw, prompt=prompt)  # [B, D]
        return self.proj(feat)  # LazyLinear 首次前向自動決定 D→out_dim

class SmolVLM2Extractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, out_dim=512,
                 model_name="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
                 freeze=True, prompt="."):
        super().__init__(observation_space, features_dim=out_dim)
        self.vla = SmolVLM2Encoder(model_name=model_name, out_dim=384, freeze=freeze)
        self.prompt = prompt
        state_dim = observation_space["state"].shape[0]
        self.mlp = nn.Sequential(
            nn.Linear(384 + state_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, obs):
        img = obs["image"].to(dtype=th.float32)
        if img.ndim == 4 and img.shape[-1] in (1, 3, 4):
            img = img.permute(0, 3, 1, 2)
        img = img / 255.0
        state = obs["state"].to(dtype=th.float32)
        img_feat = self.vla(img, prompt=self.prompt)
        return self.mlp(th.cat([img_feat, state], dim=1))