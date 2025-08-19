# RL + SmolVLM2 專案說明（中文版）

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

## 📦 專案架構說明

```
RL_code_with_SmolVLM2/
├── train.py                 # 訓練主程式
├── eval.py                  # 評估腳本
├── callbacks.py             # TensorBoard 與 log callback
│
├── envs/                    # 環境模組
│   └── xarm6_env.py         # ROS2 控制與 gym 環境
│
├── objects/                 # 模擬物件模組
│   ├── sim_object.py        # 物件資料結構與互動
│   └── spawner.py           # 物件產生器（呼叫 spawn service）
│
├── reward/                  # reward 函數設計
│   └── reward.py
│
├── vlm/                     # VLM 模組
│   ├── sync_api.py          # VLM 同步呼叫 API（回傳 score）
│   └── vlm_core.py          # 內部實作與快取
│
├── models/                  # SmolVLM2 特徵抽取器
│   └── feature_extractor.py
│
└── rl_utils/                # 公用模組（取代原 utils/）
    ├── image.py             # 影像處理工具（resize 等）
    └── gripper_control.py   # 夾爪附著與分離控制（attach/detach）
```

---

## 🧠 系統架構與訓練流程

1. **特徵抽取（Feature Extractor）**
   - 使用 `SmolVLM2`，將影像與 prompt 一起餵給模型，產生語義特徵向量。
   - 整合 joint state 形成最終的 observation。

2. **強化學習訓練（SAC）**
   - 使用 `state + image` 作為觀測，從 `SAC` 進行訓練。
   - reward 結合三部份：幾何距離、版面整齊度、VLM 評分。

3. **物件管理與互動**
   - `sim_object.py`: 表示一個物件的狀態與位置。
   - `spawner.py`: 呼叫 ROS2 `/spawn_entity` 將物件加進 Gazebo 模擬場景。

4. **夾爪控制與附著**
   - 使用 Link Attacher Service：`/ATTACHLINK`, `/DETACHLINK` 控制物件抓取與放下。

5. **評估（eval.py）**
   - 載入最佳模型並執行若干回合，紀錄：
     - 每回合 reward
     - 幾何 + VLM 版面評分

---

## ⚙️ 設定與執行

執行訓練：
```bash
python train.py
```

執行評估：
```bash
python eval.py
```

設定（環境變數或直接修改 `train.py`）：
- `VLM_INTERVAL`：每幾步送一張圖給 VLM 評分
- `VLM_PROMPT`：VLM 的指引語句，如「桌面整齊排列、間距一致...」

---

## ✅ 注意事項

- 若出現 ROS2 publisher warning，無須擔心。
- `VecTransposeImage` warning 是正常現象，因為 eval 環境未被自動 wrap。
- 請確保所有模型與 topic 名稱與你在 `URDF/SDF` 以及 ROS2 設定相符。

---

若你想繼續發展，可考慮：
- 將 VLM 部分改為 async 呼叫（替換 `sync_api`）
- 強化 spawn 隨機化，或納入更複雜物件種類
- 改善 reward 設計與穩定性

---

若有任何問題或需要程式碼說明，歡迎隨時來問 🐣

