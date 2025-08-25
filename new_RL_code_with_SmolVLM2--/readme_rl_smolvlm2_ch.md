# RL_code_with_SmolVLM2 — 中文說明

本專案目標是建構一套語言引導的強化學習系統，讓機械手臂能根據視覺語言模型（VLM）的反饋來「整理桌面」，將物品擺放得更整齊、有序。

此專案結合：
- 機械手臂控制（xArm6 + Gazebo 模擬）
- 強化學習（SAC 演算法）
- 語言引導的美學判斷（SmolVLM2）

---

## 🎯 研究主題：

**Learning to Tidy Up: Language Model-Guided Aesthetic Optimization for Robotic Tabletop Arrangement**

### 📄 參考論文：
- RL-VLM-F: Reinforcement Learning from Vision-Language Foundation Model Feedback

本專案略過小模型訓練，直接由 VLM 提供評分作為強化學習 reward。

---

## 📦 專案結構

```
RL_code_with_SmolVLM2/
├── train.py                # 訓練入口點
├── eval.py                 # 模型評估腳本
├── callbacks.py            # TensorBoard 與日誌記錄
│
├── envs/                   # 自訂 Gymnasium 環境
│   ├── xarm6_env.py
│
├── objects/                # 模擬物件與生成工具
│   ├── sim_object.py
│   └── spawner.py
│
├── reward/                 # 獎勵函式
│   └── reward.py
│
├── vlm/                    # 視覺語言模型 (VLM) 介面
│   ├── sync_api.py         # 同步 VLM 評分
│   └── vlm_core.py         # 快取介面，可替換為實體模型
│
├── models/                 # SAC 用的特徵提取器
│   └── feature_extractor.py
│
└── rl_utils/               # 工具函式（圖像處理、夾爪控制）
    ├── image.py
    └── gripper_control.py
```

---

## 🧠 功能特色

### 環境（`envs/xarm6_env.py`）
- 相容 Gymnasium API
- 使用 ROS 2 控制 xArm6 關節與夾爪
- 自動模擬物件的黏貼／釋放
- 支援 VLM 評分（排列整齊度）
- 模組化設計可替換 reward 計算

### VLM 整合
- 可透過 `VLM_INTERVAL` 設定多久送一次影像給 VLM
- VLM 分數範圍 0-1，越高代表越整齊
- 可自訂提示語句，如：`"桌面物品整齊排列、等間距、邊緣對齊。"`

### 獎勵函式（`reward/reward.py`）
- `geom_reward`：物體距離 + 懲罰區域
- `layout_score`：整齊度（排列、對齊）
- `final_reward`：綜合權重，可加入 VLM 評分

### 訓練流程（`train.py`）
- 使用 Stable-Baselines3 的 SAC 演算法
- 搭配 SmolVLM2 當作特徵提取器
- 使用 VecEnv 包裝環境 + 監控記錄
- 支援 TensorBoard 可視化、模型儲存

### 模型評估（`eval.py`）
- 載入已訓練模型，進行多次測試
- 顯示 reward / layout / VLM 分數統計

### 物體模擬
- `SimObject`：追蹤物體位置與黏貼狀態
- `Spawner`：產生 URDF 物件於 Gazebo 模擬器

---

## 🛠 執行注意事項

### ROS 2 警告
如出現：
```
Publisher already registered for provided node name...
```
這是正常的 node 名稱重複註冊，功能不受影響。

### Eval Callback 警告
```
Training and eval env are not of the same type
```
這是由 VecTransposeImage 導致的格式不一致，也不影響功能。

---

## 📈 使用 TensorBoard 觀看訓練結果

訓練日誌會儲存在 `runs/xarm6_train/` 目錄下。

啟動方式：
```bash
tensorboard --logdir=runs/xarm6_train --port=6006
```
然後打開瀏覽器輸入：
```
http://localhost:6006
```

### 可視化指標：
- `Episode/Reward`：每個 episode 的總 reward
- `Metrics/VLM_score`：VLM 回饋分數
- `Metrics/Layout`：物件整齊度評分
- `Metrics/Gripper`：夾爪使用次數
- `Actions/all` 與 `Actions/gripper`：動作直方圖
- 預覽影像：`runs/xarm6_train/samples/`

如果尚未安裝 TensorBoard：
```bash
pip install tensorboard
```

---

## 🧪 評估模型效能

```bash
python eval.py
```
會自動載入最佳模型並評估整體表現。

---

## 💡 未來擴充方向
- 將 sync VLM 改為 async 以加快推論速度
- 多物件 layout reward 強化
- 導出評估結果為 CSV 報表
- 接入真實的 VLM API 模型

---

## 📋 總結
這套框架可以讓你：
- 訓練 RL agent 整理桌面物品
- 透過 layout + VLM 評分進行學習強化
- 測試與評估策略效果（整齊度、自主操作）

模組化的設計讓後續研究如 reward 設計、模型替換、實體機實驗變得更簡單。

