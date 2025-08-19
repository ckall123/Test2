# RL_code_with_SmolVLM2

A structured reinforcement learning (RL) framework for robotic tabletop object rearrangement using ROS 2, SmolVLM2 vision-language model, and Stable-Baselines3.

---

## 📦 Project Structure

```
RL_code_with_SmolVLM2/
├── train.py                # Training entrypoint
├── eval.py                 # Evaluation script
├── callbacks.py            # TensorBoard + metric logging
│
├── envs/                   # Custom Gymnasium environment
│   ├── xarm6_env.py
│
├── objects/                # Simulated objects & spawning logic
│   ├── sim_object.py
│   └── spawner.py
│
├── reward/                 # Reward functions
│   └── reward.py
│
├── vlm/                    # Vision-Language Model interface
│   ├── sync_api.py         # VLM scoring (sync)
│   └── vlm_core.py         # Cached VLM interface (mockable)
│
├── models/                 # Feature extractors for SAC
│   └── feature_extractor.py
│
└── rl_utils/               # Utilities (image, gripper control)
    ├── image.py
    └── gripper_control.py
```

---

## 🔧 Features

### Environment (`envs/xarm6_env.py`)
- Compatible with Gymnasium API
- ROS 2 powered xArm6 joint & gripper control
- Automatic attach/detach simulation for object grasping
- VLM evaluation support (layout scoring)
- Modular reward design (geom/layout/VLM)

### VLM Integration
- Periodically (configurable `VLM_INTERVAL`) sends downsampled image to `sync_api.get_vlm_score`
- Score (0-1) reflects how well objects are arranged
- Prompt is configurable: `"桌面物品整齊排列、等間距、邊緣對齊。"`

### Rewards (`reward/reward.py`)
- `geom_reward`: closeness to object, with workspace penalties
- `layout_score`: alignment + spacing among objects
- `final_reward`: weighted sum with optional VLM score boost

### Training (`train.py`)
- Uses Stable-Baselines3 SAC + custom feature extractor (SmolVLM2)
- Environment wrapped in Monitor + TimeLimit + DummyVecEnv
- TensorBoard logging (reward, gripper, actions, VLM layout scores)
- Automatic checkpointing & best model saving

### Evaluation (`eval.py`)
- Loads trained model and runs multiple episodes
- Logs and prints reward + layout + VLM statistics

### Object Simulation
- `SimObject`: track object position, link, and attachment
- `Spawner`: spawn URDF objects into Gazebo from model database

---

## 🛠 Execution Notes

### ROS 2 Warning
You may see:
```
Publisher already registered for provided node name...
```
This is safe and caused by node name reuse. You can ignore it if functionality is normal.

### Eval Callback Warning
```
Training and eval env are not of the same type
```
This is safe: it comes from `VecTransposeImage` used by Stable-Baselines3. If you don't explicitly use image transposition, you can suppress this.

---

## 🧠 Future Improvements
- Switch to `async_api.py` for parallel VLM inference
- Multi-object layout planning reward
- Full evaluation script with CSV export
- Replace mocked VLM with real API call

---

## 📋 Summary
This codebase allows you to:
- Train an RL agent to rearrange objects on a table
- Score progress with layout/VLM-based metrics
- Evaluate agent policies with layout-aware vision models

Clean modular structure ensures that future research (e.g. new reward signals, better VLMs, more complex scenes) is easy to prototype.

