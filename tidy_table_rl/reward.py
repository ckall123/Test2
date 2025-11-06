import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from typing import Tuple
import numpy as np


class RewardModel(nn.Module):
    def __init__(self, backbone: str = "resnet18"):
        super().__init__()
        if backbone == "resnet18":
            base = models.resnet18(pretrained=False)
            self.feature_extractor = nn.Sequential(*list(base.children())[:-1])  # 去掉最後的 fc
            self.head = nn.Linear(512, 1)
        else:
            raise ValueError("Unsupported backbone")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.feature_extractor(x).squeeze(-1).squeeze(-1)
        reward = self.head(feat)
        return reward.squeeze(-1)  # (B)


def preprocess_image(img: np.ndarray, size: Tuple[int, int] = (224, 224)) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(img)


def pairwise_loss(reward_a: torch.Tensor, reward_b: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    # Bradley-Terry: sigmoid(r_a - r_b)
    logit = reward_a - reward_b
    return F.binary_cross_entropy_with_logits(logit, label.float())


# ★★★ Load / Save Utilities ★★★
def save_model(model: RewardModel, path: str):
    torch.save(model.state_dict(), path)


def load_model(path: str, device: str = "cpu") -> RewardModel:
    model = RewardModel()
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model
