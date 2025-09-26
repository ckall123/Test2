from __future__ import annotations
import time, cv2, numpy as np
import gymnasium as gym
from gymnasium import spaces

import rclpy
from scipy.spatial.transform import Rotation as R

from env.camera import TopDownCamera            # 獲取上視角 RGB 觀測
from env.moveit_controller import MoveIt2Controller  # 控制 MoveIt2 的機械手臂與夾爪
from env.gripper_contact import GripperContact       # 夾爪是否同時接觸物體
from env.attach_detach import AttachDetachHelper     # 模擬 attach/detach 機制
from object.spawner import Spawner                   # 隨機生成物品
from env import reward as R                          # 套件化 reward 設計（align + VLM）

class XArm6TidyEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self,
                 img_h: int = 96, img_w: int = 96,
                 max_steps: int = 80,
                 step_dt: float = 0.8,
                 pos_scale: float = 0.03,
                 rot_scale: float = 0.15,
                 use_vlm_end: bool = True,
                 objects_n: int = 4):
        """
        初始化環境
        - img_h/w: 相機觀測解析度
        - max_steps: 每回合最大步數
        - step_dt: 每步執行時間（秒）
        - pos_scale/rot_scale: 動作比例縮放
        - use_vlm_end: 是否於 episode 結尾額外給 VLM 分數
        - objects_n: 每次生成的物品數
        """
        super().__init__()
        self.img_h, self.img_w = img_h, img_w
        self.max_steps = max_steps
        self.step_dt = step_dt
        self.pos_scale = pos_scale
        self.rot_scale = rot_scale
        self.use_vlm_end = use_vlm_end
        self.objects_n = objects_n

        if not rclpy.ok():
            rclpy.init()
        self.camera = TopDownCamera()
        self.ctrl = MoveIt2Controller()
        self.contact = GripperContact.get_instance()
        self.attacher = AttachDetachHelper()
        self.spawner = Spawner()

        self.ee_pos = np.zeros(3, dtype=np.float32)                  # End-effector position
        self.ee_quat = np.array([0, 0, 0, 1], dtype=np.float32)     # End-effector rotation

        self.observation_space = spaces.Box(0, 255, shape=(img_h, img_w, 3), dtype=np.uint8)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)  # Δpos(3) + Δrot(3) + gripper(1)

        self.step_count = 0
        self.spawned = []

    def _home(self):
        """回到初始位置並張開夾爪"""
        self.ctrl.move_arm_home()
        self.ctrl.open_gripper()

    def _frame(self) -> np.ndarray:
        """擷取與處理相機圖像（resize）"""
        img = self.camera.get_latest_frame()
        if img is None or img.size == 0:
            img = np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8)
        return cv2.resize(img, (self.img_w, self.img_h), interpolation=cv2.INTER_AREA).astype(np.uint8)

    def _apply(self, dpos, drot, g_open):
        """
        套用動作至機械手臂
        - dpos: 位置偏移
        - drot: 歐拉角偏移
        - g_open: gripper 開合程度（0~1）
        """
        tgt_pos = self.ee_pos + dpos
        delta_rot = R.from_euler('xyz', drot)
        curr_rot = R.from_quat(self.ee_quat)
        tgt_quat = (delta_rot * curr_rot).as_quat()

        success = self.ctrl.move_to_pose(tgt_pos, tgt_quat, duration=self.step_dt)
        if success:
            self.ee_pos = tgt_pos
            self.ee_quat = tgt_quat

        self.ctrl.control_gripper(g_open, duration=0.3)

        # 如果夾住並雙邊接觸 → 附著
        if g_open < 0.5:
            if self.contact.dual_contact():
                self.attacher.attach()
        else:
            if self.attacher.is_attached():
                self.attacher.detach()

    def _dense_step_reward(self, img: np.ndarray) -> tuple[float, dict]:
        """計算對齊 reward（非 VLM）"""
        r_align = float(R.reward_alignment(img))
        return r_align, {"align": r_align}

    def reset(self, *, seed=None, options=None):
        """環境重置"""
        super().reset(seed=seed)
        self.step_count = 0
        self.spawner.delete_all()
        self.spawned = self.spawner.spawn_random_objects(count=self.objects_n)
        self._home()

        pos, quat = self.ctrl.get_current_pose()
        if pos is not None:
            self.ee_pos = np.array(pos, dtype=np.float32)
            self.ee_quat = np.array(quat, dtype=np.float32)
        return self._frame(), {"spawned": len(self.spawned)}

    def step(self, action: np.ndarray):
        """主要 RL 執行流程，每一步的環境互動與 reward 計算"""
        self.step_count += 1
        a = np.clip(action, -1.0, 1.0)
        dpos = a[:3] * self.pos_scale
        drot = a[3:6] * self.rot_scale
        g_open = (a[6] + 1.0) * 0.5

        self._apply(dpos, drot, g_open)
        time.sleep(self.step_dt)

        img = self._frame()
        r_dense, info = self._dense_step_reward(img)

        terminated = r_dense > 0.85
        truncated = self.step_count >= self.max_steps

        # 若結束 → 啟用 VLM 判分
        if (terminated or truncated) and self.use_vlm_end:
            r_vlm = float(R.reward_vlm(img, "align objects in a neat row"))  # vlm
            info["vlm"] = r_vlm
            r_dense = 0.7 * r_dense + 0.3 * r_vlm
            if r_vlm > 0.8:
                info["success"] = True

        return img, float(r_dense), terminated, truncated, info

    def render(self):
        return self._frame()

    def close(self):
        """關閉 ROS node"""
        try:
            self.camera.destroy_node()
            self.ctrl.destroy_node()
            self.spawner.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()
