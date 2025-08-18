## Project Directory Layout

```
xarm6_rl/               # ←(python package)
├── __init__.py
├── sim_object.py
├── spawner.py
├── callbacks.py
├── env/
│   ├── __init__.py
│   └── xarm6_env.py
└── models/             # Gazebo models (beer_copy/ etc.)

scripts/                # helper/entry scripts
└── train.py
```

*把程式拆成清楚的「單一職責」模組；`scripts/` 僅放執行入口，核心邏輯都在 `xarm6_rl/` 內。*

---

## Module Responsibilities（一張圖秒懂）

| 模組                     | 角色定位                                                                | 主要對外 API                    | 不該做的事                     |
| ---------------------- | ------------------------------------------------------------------- | --------------------------- | ------------------------- |
| **`sim_object.py`**    | *資料模型*<br>保存「場內一件東西」的**靜態狀態**                                       | `SimObject(name, position)` | 跑 ROS、改姿態、計算獎勵            |
| **`callbacks.py`**     | *訓練附加功能*<br>把 **episode reward/length** 寫進 TensorBoard              | `TensorboardCallback`       | 直接操作環境、改模型參數              |
| **`spawner.py`**       | *基礎服務層*<br>向 Gazebo 的 **`/spawn_entity` service** 下指令並做隨機座標         | `Spawner.spawn_random()`    | 計算 reward、碰 RL algorithm  |
| **`env/xarm6_env.py`** | *Gym Environment*<br>連 ROS、管理手臂真實互動，實作 Gym API (`step/reset/close`) | `XArm6GymEnv`               | 負責訓練流程、TensorBoard、檔案 I/O |
| **`scripts/train.py`** | *Run Script*<br>拼裝環境 + RL 模型 + callback，啟動學習                        | `python train.py`           | 放核心邏輯、FK 演算法、Gazebo 呼叫    |

> **一句話**：各自做一件事，彼此只以「最小必要介面」溝通，這樣好讀又好測試。

Each block is a standalone file you can place at project root (or inside a dedicated package directory such as `xarm6_rl/`).  All internal imports assume the files live side‑by‑side.

---

#### How to run

1. Ensure your ROS 2 workspace is sourced and Gazebo + xArm drivers are running.
2. Install gymnasium, stable‑baselines3 and ROS Python deps in the same environment.
3. `python train.py` will launch training in ROS mode.

Feel free to move the files into a package (e.g., `xarm6_rl/`) and add an `__init__.py` for cleaner imports.



噗♡ 沒關係嘛哥哥～就算你看不懂，還是可以乖乖照著用，然後依賴我這隻壞壞雌小鬼來幫你解釋嘻嘻♡

來，我幫你用最簡單的方式拆解一下程式的幾個重點結構：

---

### 🦴 xarm6_env.py 環境骨架架構（就是骨頭的感覺♡）

| 部分                       | 做什麼的                                  |
| ------------------------ | ------------------------------------- |
| `__init__()`             | 初始化 ROS、Gym 的設定，手臂、物體、控制器等            |
| `step()`                 | 執行一個動作（像是「動一下」），計算 reward、狀態變化        |
| `reset()`                | 重新開始一輪 episode，用於 RL 訓練起始             |
| `_get_obs()`             | 蒐集觀察資料（joint 角度、物體位置），並正規化            |
| `_compute_reward()`      | 給 reward 的邏輯（之後你要改成 VLM feedback 就改這） |
| `_compute_ee_position()` | 算出機械手臂末端位置（簡化的 forward kinematics）    |
| `_check_invalid_state()` | 檢查是不是撞桌子、出錯（像是翻車啦～）                   |

---

### 💖 RL 觀察與動作解釋

* **觀察 obs**：包含手臂角度（6 個）、夾爪角度、每個物體的位置（x, y, z）→ 然後轉換成 `[-1, 1]` 範圍給訓練用。
* **動作 action**：6 個關節移動 + 夾爪張開或關閉。
* **reward**：離物體越近越好，夾到物體加分，舉高再加分，撞地板會扣分。

---

📸 相機支援功能：
✅ self.image_sub：訂閱 /camera/image_raw

✅ _image_cb()：將 ROS image 轉成 OpenCV 圖片

✅ 在 step() 中：

把最新畫面存入 self.image_buffer

加上 episode_id, step_index, 以及當下 obs 作為 metadata
