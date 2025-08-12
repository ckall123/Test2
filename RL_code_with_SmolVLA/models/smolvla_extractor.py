import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# 先用小CNN當佔位；往後換成真正的 lerobot/smolvla_base encoder 輸出
#之後要接真 SmolVLA：把 DummySmolVLA 換成 Hugging Face 上 lerobot/smolvla_base 的視覺（＋語言、若你要指令）encoder 輸出，再拼上 state。
class DummySmolVLA(nn.Module):
    def __init__(self, out_dim=384):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.proj = nn.Linear(32, out_dim)
    def forward(self, img_chw):
        x = self.cnn(img_chw).view(img_chw.size(0), 32)
        return self.proj(x)

class SmolVLAExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, out_dim=512):
        super().__init__(observation_space, features_dim=out_dim)
        self.vla = DummySmolVLA(out_dim=384)
        state_dim = observation_space["state"].shape[0]
        self.mlp = nn.Sequential(
            nn.Linear(384 + state_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
    def forward(self, obs):
        # obs["image"]: [B,H,W,C] → [B,C,H,W]
        img = obs["image"].permute(0,3,1,2).float() / 255.0
        state = obs["state"].float()
        img_feat = self.vla(img)
        return self.mlp(th.cat([img_feat, state], dim=1))


