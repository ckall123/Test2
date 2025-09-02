# =========================
# FILE: models/feature_extractor.py
# =========================
from __future__ import annotations
import warnings
import torch as th
import torch.nn as nn

# 先嘗試使用 LeRobot 的 SmolVLA；失敗再回退到 SigLIP
_HAS_LEROBOT = False
try:
    from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # type: ignore
    _HAS_LEROBOT = True
except Exception:
    _HAS_LEROBOT = False

# SigLIP（transformers 原生，穩定）
try:
    from transformers import SiglipImageProcessor, SiglipVisionModel
    _HAS_SIGLIP = True
except Exception:
    _HAS_SIGLIP = False

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class _SmolVLABackbone(nn.Module):
    """
    嘗試用 SmolVLAPolicy 取出上下文特徵。
    注意：LeRobot API 變動快，這裡做了多重 fallback，
    如果某個介面不存在就回退到簡單平均池化做 1D 向量。
    """
    def __init__(self, model_name: str = "lerobot/smolvla_base", freeze: bool = True, device: str | None = None):
        super().__init__()
        if not _HAS_LEROBOT:
            raise RuntimeError("lerobot 未安裝或版本不符，無法載入 SmolVLA。請先 `pip install -U lerobot` 並確保 `pip install -e \".[smolvla]\"`")
        self.policy = SmolVLAPolicy.from_pretrained(model_name)
        self.device = th.device(device or ("cuda" if th.cuda.is_available() else "cpu"))
        self.policy.to(self.device)
        if freeze:
            for p in self.policy.parameters():
                p.requires_grad = False
        self.policy.eval()

        # 嘗試推斷特徵維度（以 config 或內部 hidden size 為準；失敗就設 512）
        feat_dim = 512
        for key in ("hidden_size", "d_model", "dim", "vision_hidden_size"):
            dim = getattr(getattr(self.policy, "config", self.policy), key, None)
            if isinstance(dim, int):
                feat_dim = dim
                break
        self.out_dim = feat_dim
        # 當我們只能拿到 token map 時，用一個 Lazy Linear 做降維
        self.fallback_proj = nn.Sequential(nn.LazyLinear(self.out_dim), nn.ReLU())

    @th.inference_mode()
    def forward(self, imgs_bchw: th.Tensor, state: th.Tensor, text: str = ".") -> th.Tensor:
        """
        imgs_bchw: float in [0,1], shape [B,3,H,W]
        state: [B, Ds]
        回傳: [B, D] 1D 向量特徵
        """
        B = imgs_bchw.shape[0]
        device = self.device
        imgs_bchw = imgs_bchw.to(device)
        state = state.to(device)

        # 嘗試各種常見 API 取「上下文特徵」
        # 1) 官方可能提供的 encode_context / get_context_features
        for meth in ("encode_context", "get_context_features", "encode"):
            if hasattr(self.policy, meth):
                try:
                    out = getattr(self.policy, meth)(images=imgs_bchw, state=state, text=[text] * B)  # type: ignore
                    if isinstance(out, th.Tensor) and out.dim() == 2:
                        return out
                    if isinstance(out, th.Tensor) and out.dim() == 3:
                        # [B, T, D] -> 取 mean
                        return out.mean(dim=1)
                except Exception:
                    pass

        # 2) 走底層 .model.vlm_with_expert（可能回傳 prefix/suffix token）
        try:
            m = getattr(self.policy, "model")
            vwe = getattr(m, "vlm_with_expert")
            if hasattr(vwe, "forward"):
                # 許多實作會回傳 ((prefix_out, suffix_out), caches)
                outputs = vwe.forward(images=imgs_bchw, state=state, text=[text] * B)  # type: ignore
                if isinstance(outputs, tuple) and len(outputs) >= 1:
                    xy = outputs[0]
                    # 選一個 token 張量做池化
                    if isinstance(xy, tuple) and len(xy) >= 1 and isinstance(xy[0], th.Tensor):
                        tokens = xy[0]  # [B, T, D]
                        if tokens.dim() == 3:
                            return tokens.mean(dim=1)
        except Exception:
            pass

        # 3) 最後退：把任何可得的張量拼起來 -> 投影
        try:
            ts = []
            for name, mod in self.policy.named_modules():
                # 嘗試抓最末端有 'vision' 關鍵字的 buffer/屬性
                for attr in ("last_hidden_state", "hidden_states", "features", "embeddings"):
                    v = getattr(mod, attr, None)
                    if isinstance(v, th.Tensor):
                        ts.append(v)
            if len(ts) > 0:
                z = th.cat([t.flatten(1) for t in ts if t.numel() > 0], dim=1)
                return self.fallback_proj(z)
        except Exception:
            pass

        # 萬一什麼都沒有，就回零向量避免炸掉
        warnings.warn("[SmolVLABackbone] 回退為零向量（未能取得特徵）")
        return th.zeros((B, self.out_dim), device=device, dtype=imgs_bchw.dtype)


class _SigLIPBackbone(nn.Module):
    """
    穩定回退：使用 transformers 的 SigLIP 視覺編碼器做特徵。
    """
    def __init__(self, model_name: str = "google/siglip-base-patch16-224", freeze: bool = True, device: str | None = None):
        super().__init__()
        if not _HAS_SIGLIP:
            raise RuntimeError("transformers 未安裝或版本不含 SigLIP，請先 `pip install -U transformers`")
        self.processor = SiglipImageProcessor.from_pretrained(model_name)
        self.backbone = SiglipVisionModel.from_pretrained(model_name)
        self.device = th.device(device or ("cuda" if th.cuda.is_available() else "cpu"))
        self.backbone.to(self.device)
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.backbone.eval()
        self.out_dim = self.backbone.config.hidden_size
        self.pool = nn.AdaptiveAvgPool1d(1)

    @th.inference_mode()
    def forward(self, imgs_bchw: th.Tensor, state: th.Tensor, text: str = ".") -> th.Tensor:
        # 轉成 [B,H,W,3] np 給 processor
        imgs = (imgs_bchw.clamp(0, 1) * 255).to(th.uint8).permute(0, 2, 3, 1).cpu().numpy()
        inputs = self.processor(images=list(imgs), return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)  # [B, 3, 224, 224]
        out = self.backbone(pixel_values=pixel_values)
        # last_hidden_state: [B, N, D]，做 mean pool
        z = out.last_hidden_state.transpose(1, 2)  # [B, D, N]
        z = self.pool(z).squeeze(-1)               # [B, D]
        return z


class SmolVLAExtractor(BaseFeaturesExtractor):
    """
    SB3 的 features_extractor
    輸入 obs={"image": uint8[B,H,W,3 or C,H,W], "state": float[B,D]} → feature 向量
    - 優先用 SmolVLA (LeRobot)；若不可用則回退 SigLIP。
    """
    def __init__(
        self,
        observation_space,
        out_dim: int = 512,
        model_name: str = "lerobot/smolvla_base",
        freeze: bool = True,
        prompt: str = ".",
        device: str | None = None,
    ):
        super().__init__(observation_space, features_dim=out_dim)
        self.prompt = prompt
        self.device = device or ("cuda" if th.cuda.is_available() else "cpu")

        # backbone 選擇
        self._use_smolvla = False
        backbone: nn.Module
        try:
            backbone = _SmolVLABackbone(model_name=model_name, freeze=freeze, device=self.device)
            self._use_smolvla = True
        except Exception as e:
            warnings.warn(f"[SmolVLAExtractor] 無法使用 SmolVLA：{e}\n改用 SigLIP 回退以確保可訓練。")
            backbone = _SigLIPBackbone(model_name="google/siglip-base-patch16-224", freeze=freeze, device=self.device)

        self.backbone = backbone
        state_dim = int(observation_space["state"].shape[0])

        # 把 (image_feat, state) 拼接後投影到 out_dim
        in_dim = getattr(self.backbone, "out_dim", 512) + state_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, obs):
        img = obs["image"].to(dtype=th.float32)
        # 接受 [B,H,W,3] 或 [B,3,H,W]
        if img.ndim == 4 and img.shape[-1] in (1, 3, 4):
            img = img.permute(0, 3, 1, 2)
        # 正規化到 [0,1]
        if img.max() > 1.5:
            img = img / 255.0

        state = obs["state"].to(dtype=th.float32)
        with th.no_grad():
            img_feat = self.backbone(img, state, text=self.prompt)
        return self.mlp(th.cat([img_feat, state], dim=1))
