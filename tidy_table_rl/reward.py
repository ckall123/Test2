import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights # 導入 ResNet 權重
import numpy as np
import cv2 # 測試區塊會使用


class RewardModel(nn.Module):
    def __init__(self, backbone='resnet18'):
        super().__init__()
        
        # 專注於 ResNet18 (移除 if/else，使結構更簡潔)
        if backbone != 'resnet18':
            print(f"警告: 獎勵模型目前僅支援 resnet18，將使用預設值。")
        
        weights = ResNet18_Weights.IMAGENET1K_V1
        self.encoder = models.resnet18(weights=weights)
        
        # 將最後的全連接層替換為單一輸出 (獎勵分數)
        in_features = self.encoder.fc.in_features
        self.encoder.fc = nn.Linear(in_features, 1)

    def forward(self, x):
        # 使用 sigmoid 將輸出壓縮到 (0, 1) 區間，作為獎勵分數
        return torch.sigmoid(self.encoder(x)).squeeze(1)


def compute_bt_loss(r0, r1, y):
    # y: (batch,) 偏好標籤 (0 或 1)，表示 r1 是否優於 r0
    # r0, r1: reward scores for sigma_0 and sigma_1 (batch,)
    
    # Bradley-Terry Loss 的核心是 r1 - r0 的 logits
    logits = r1 - r0
    labels = torch.tensor(y, dtype=torch.float32, device=logits.device)
    # 使用 BCEWithLogitsLoss (數值更穩定)
    return F.binary_cross_entropy_with_logits(logits, labels)


def make_transforms():
    # 標準的 ImageNet 預處理管道
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)), # ResNet 標準輸入尺寸
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def relabel_transitions(buffer, model, device):
    """使用當前獎勵模型重新標記經驗緩衝區中的獎勵分數"""
    model.eval()
    new_rewards = []
    with torch.no_grad():
        for transition in buffer:
            img = transition['image']  # np.ndarray (H,W,3)
            tensor = make_transforms()(img).unsqueeze(0).to(device)
            r = model(tensor).item()
            transition['reward'] = r
            new_rewards.append(r)
    return new_rewards


if __name__ == '__main__':
    from PIL import Image
    # 這裡的程式碼是純粹的步驟邏輯展示 (無錯誤處理，執行失敗會直接崩潰)
    
    # 步驟 1: 讀取圖片 (cv2.imread 預設為 BGR 格式)
    dummy_img_bgr = cv2.imread("topdown_view.jpg") 
    
    # 步驟 2: 轉換圖片為 RGB (PyTorch/PIL 期望的格式)
    dummy_img_rgb = cv2.cvtColor(dummy_img_bgr, cv2.COLOR_BGR2RGB)

    # 步驟 3: 初始化獎勵模型 (載入預訓練權重)
    model = RewardModel()
    model.eval()

    # 步驟 4: 建立預處理轉換管道
    transform = make_transforms()
    
    # 步驟 5: 執行預處理、增加 batch 維度 (unsqueeze)
    x = transform(dummy_img_rgb).unsqueeze(0)
    
    # 步驟 6: 執行模型推理並打印獎勵分數
    print("Reward Score:", model(x).item())
