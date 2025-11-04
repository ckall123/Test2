import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
import numpy as np
import cv2


class RewardModel(nn.Module):
    """rψ：從影像預測原始分數（未 sigmoid）"""
    def __init__(self, backbone='resnet18'):
        super().__init__()
        if backbone != 'resnet18':
            print("警告: 目前僅支援 resnet18，已自動使用預設值。")

        weights = ResNet18_Weights.IMAGENET1K_V1
        self.encoder = models.resnet18(weights=weights)
        in_features = self.encoder.fc.in_features
        self.encoder.fc = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.encoder(x).squeeze(1)


def compute_bt_loss(r0, r1, y):
    logits = r1 - r0
    labels = torch.tensor(y, dtype=torch.float32, device=logits.device)
    return F.binary_cross_entropy_with_logits(logits, labels)


def make_transforms():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


def relabel_transitions(buffer, model, device, diff_mode=True):
    """
    使用 rψ 重新標記 buffer reward。
    diff_mode=True → 使用 r(next_image) - r(image)
    """
    model.eval()
    transform = make_transforms()
    new_rewards = []

    with torch.no_grad():
        for transition in buffer:
            img = transform(transition['image']).unsqueeze(0).to(device)

            if diff_mode and 'next_image' in transition:
                img_next = transform(transition['next_image']).unsqueeze(0).to(device)
                r = (model(img_next) - model(img)).item()
            else:
                r = model(img).item()

            transition['reward'] = r
            new_rewards.append(r)

    return new_rewards


if __name__ == '__main__':
    dummy_img_bgr = cv2.imread("topdown_view.jpg")
    dummy_img_rgb = cv2.cvtColor(dummy_img_bgr, cv2.COLOR_BGR2RGB)

    model = RewardModel()
    model.eval()
    transform = make_transforms()

    x = transform(dummy_img_rgb).unsqueeze(0)
    print("Raw Reward Score:", model(x).item())
