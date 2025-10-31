#!/usr/bin/env python3
"""
xarm6_gym_env.py (RL-VLM-F 架構)

職責:
1. 實現 Gymnasium API (reset, step)。
2. 整合底層 ROS 2 工具 (MoveIt, 相機, 姿態追蹤, 接觸, 生成器)。
3. 在 'step' 中固定回傳 reward = 0.0。
4. 在 'info' 字典中回傳當前的高品質影像，供 ReplayBuffer 和 VLM 標註使用。
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass
from typing import Dict, Tuple, Any, List

# --- 匯入階段一的底層工具 ---
from camera import TopDownCamera
from moveit_controller import MoveItController, ARM_JOINT_NAMES
from spawner import Spawner
from gripper_contact import ContactMonitor
from pose_tracker import PoseTracker, load_target_names
from utils import ActionHistory, preprocess_image


@dataclass
class XArmEnvConfig:
    """ 環境設定 (取代舊的 utils.EnvConfig) """
    image_size: Tuple[int, int] = (96, 96)  # Policy 觀察用的影像 (W, H)
    max_steps: int = 400
    action_scale: float = 0.08
    num_objects: int = 3  # reset 時生成的物件數量
    action_hist_len: int = 10


class XArm6Env(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, node, executor, cfg: XArmEnvConfig = XArmEnvConfig()):
        super().__init__()
        self.node = node
        self.executor = executor
        self.cfg = cfg

        # 1. 初始化底層工具
        self.controller = MoveItController(node, executor)
        self.camera = TopDownCamera(node)
        self.spawner = Spawner(node, executor)
        self.contact_monitor = ContactMonitor(node)
        self.tracker = PoseTracker(node)

        # 2. 獲取可追蹤的物件名稱
        self.object_names: List[str] = load_target_names()
        self.node.get_logger().info(f"環境將追蹤 {len(self.object_names)} 個物件: {self.object_names}")

        # 3. 動作空間 (6-DoF Arm + 1-DoF Gripper)
        self.act_dim = len(ARM_JOINT_NAMES) + 1
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.act_dim,), dtype=np.float32)

        # 4. 觀測空間 (Policy $\pi_\theta$ 用的)
        # "image": 縮放過的影像
        # "state": 機器人狀態 + 物件姿態 + 動作歷史
        w, h = self.cfg.image_size
        image_shape = (h, w, 3)  # HWC
        
        self.action_hist = ActionHistory(self.cfg.action_hist_len, action_dim=self.act_dim)
        self.hist_dim = self.cfg.action_hist_len * self.act_dim

        # 計算 state 維度:
        # 6 (關節) + 1 (夾爪) + 1 (夾爪接觸) + 3*N (N個物件的xyz) + N (N個物件的yaw) + hist_dim
        self.state_dim = 6 + 1 + 1 + (len(self.object_names) * 3) + len(self.object_names) + self.hist_dim
        
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=255, shape=image_shape, dtype=np.uint8),
            "state": spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32),
        })

        self.step_count: int = 0
        self.node.get_logger().info("✅ XArm6Env (RL-VLM-F) 初始化完成")

    # ============== Gym API (核心) ==============

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        """
        重置環境:
        1. (Hard Reset) 清空並重新生成物件。
        2. 手臂歸位，夾爪打開。
        3. 回傳 (obs, info)，info 包含高品質影像。
        """
        super().reset(seed=seed)
        self.node.get_logger().info("--- Env Reset ---")
        self.step_count = 0
        self.action_hist.clear()

        # 1. 硬重置：清空桌面並重新生成
        self.spawner.delete_all()
        self.spawner.spawn_random_objects(count=self.cfg.num_objects)

        # 2. 手臂歸位，夾爪打開
        self.controller.go_home()
        self.controller.move_gripper(0.85) # 0.85 接近全開

        # 3. 獲取觀測
        # 抓取全解析度影像 (給 VLM 和 r_psi 用)
        full_img = self._grab_full_res_image()
        
        # 組合觀測 (obs)
        obs = self._get_obs(full_img)
        
        # Info 字典 (給 Buffer 儲存用)
        info = {"image": full_img}
        
        self.node.get_logger().info("--- Env Reset 完成 ---")
        return obs, info

    def step(self, action: np.ndarray):
        """
        執行一步:
        1. 套用動作 (action)。
        2. 獲取新觀測。
        3. 回傳 reward = 0.0 (等待 Relabel)。
        4. 回傳 info 包含高品質影像。
        """
        self.step_count += 1

        # 1. 執行動作並記錄歷史
        self._apply_action(action)
        self.action_hist.add(action)

        # 2. 獲取觀測
        full_img = self._grab_full_res_image()
        obs = self._get_obs(full_img)

        # 3. RL-VLM-F 核心: Reward 固定為 0.0
        reward = 0.0

        # 4. 結束條件 (僅依賴最大步數)
        terminated = False
        truncated = bool(self.step_count >= self.cfg.max_steps)

        # 5. Info 字典 (給 Buffer 儲存用)
        info = {"image": full_img}

        return obs, reward, terminated, truncated, info

    def render(self) -> np.ndarray:
        """ 回傳當前的高品質 RGB 畫面 """
        return self._grab_full_res_image()

    def close(self):
        """ 清理環境 (例如刪除所有生成的物件) """
        self.spawner.delete_all()
        self.node.get_logger().info("環境關閉，已清除所有物件。")

    # ============== 輔助函式 ==============

    def _apply_action(self, action: np.ndarray) -> None:
        """ 將 [-1, 1] 範圍的動作轉換為 MoveIt 命令 """
        a = np.asarray(action, dtype=np.float32)

        # 1) 目標關節增量 & 夾爪
        dq = a[:6] * self.cfg.action_scale
        q_now = self.controller.get_joint_positions()
        q_tar = q_now + dq
        
        # 夾爪: [-1, 1] -> [0, 0.8552]
        g_val = float(np.clip((a[-1] + 1.0) * 0.5, 0.0, 1.0)) * 0.8552
        
        # 2) 執行規劃
        try:
            self.controller.plan_and_execute(q_tar, g_val)
        except Exception as e:
            self.node.get_logger().warn(f"[apply_action] 規劃執行失敗: {e}")

        # 3) Spin ROS 讓服務回調
        try:
            for _ in range(5): # 少量 spin 確保狀態更新
                self.executor.spin_once(timeout_sec=0.01)
        except Exception:
            pass

    def _grab_full_res_image(self) -> np.ndarray:
        """ 獲取相機的原始(全解析度)影像 """
        img = self.camera.get_latest_frame(self.executor)
        return img

    def _get_obs(self, full_image: np.ndarray) -> Dict[str, Any]:
        """
        組合 Policy $\pi_\theta$ 需要的觀測:
        - image: 縮放過的影像
        - state: 狀態向量
        """
        # 1. 影像觀測 (縮放)
        image_obs = preprocess_image(full_image, self.cfg.image_size)
        
        # 2. 狀態向量觀測
        state_vec = self._build_state_features()
        
        return {
            "image": image_obs.astype(np.uint8), 
            "state": state_vec.astype(np.float32)
        }

    def _build_state_features(self) -> np.ndarray:
        """
        組裝狀態向量 (State Vector):
        [robot_state, objects_state, action_history]
        """
        # 1. 機器人狀態 (6+1+1)
        q = self.controller.get_joint_positions()
        g = np.array([self.controller.get_gripper_state()], dtype=np.float32)
        
        # 檢查是否接觸 (任意物件)
        contact_state = self.contact_monitor.in_contact(target=None)
        c = np.array([1.0 if contact_state else 0.0], dtype=np.float32)

        # 2. 物件狀態 (N*3 + N)
        obj_states = self.tracker.get_object_states(self.object_names, radius_lookup=None)
        
        obj_pos_list = []
        obj_yaw_list = []
        
        # 確保順序一致
        obj_map = {o["name"]: o for o in obj_states}
        for name in self.object_names:
            if name in obj_map:
                obj_pos_list.append(obj_map[name]["pos"])
                obj_yaw_list.append(obj_map[name]["yaw"])
            else:
                # 如果物件沒偵測到 (例如掉出)
                obj_pos_list.append(np.array([0.0, 0.0, 0.0]))
                obj_yaw_list.append(0.0)

        obj_pos_flat = np.concatenate(obj_pos_list, axis=0)
        obj_yaw_flat = np.array(obj_yaw_list, dtype=np.float32)

        # 3. 動作歷史
        hist_vec = self.action_hist.vector()
        if hist_vec.size == 0:
            hist_vec = np.zeros(self.hist_dim, dtype=np.float32)

        parts = [
            q.astype(np.float32), 
            g, 
            c, 
            obj_pos_flat.astype(np.float32), 
            obj_yaw_flat,
            hist_vec.astype(np.float32)
        ]
        
        # 檢查維度是否匹配
        state_vec = np.concatenate(parts, axis=0)
        if state_vec.shape[0] != self.state_dim:
            self.node.get_logger().error(f"狀態維度不匹配! 期望 {self.state_dim}, 得到 {state_vec.shape[0]}")
            # 發生錯誤時回傳零向量，避免崩潰
            return np.zeros(self.state_dim, dtype=np.float32)

        return state_vec