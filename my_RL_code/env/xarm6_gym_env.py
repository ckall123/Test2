# env/xarm6_gym_env.py
from __future__ import annotations
import time, math, cv2, numpy as np
import gymnasium as gym
from gymnasium import spaces

import rclpy
from env.camera import TopDownCamera
from env.joint_controller import JointController
from env.ik import solve_ik, get_current_pose
from object.spawner import Spawner
from env import reward as R

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
        super().__init__()
        self.img_h, self.img_w = img_h, img_w
        self.max_steps = max_steps
        self.step_dt = step_dt
        self.pos_scale = pos_scale
        self.rot_scale = rot_scale
        self.use_vlm_end = use_vlm_end
        self.objects_n = objects_n

        # ROS2 nodes
        if not rclpy.ok():
            rclpy.init()
        self.camera = TopDownCamera()
        self.ctrl = JointController()
        self.spawner = Spawner()

        # EE 狀態（reset 時由 TF 取一次）
        self.ee_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.ee_quat = np.array([0, 0, 0, 1], dtype=np.float32)

        # gym spaces
        self.observation_space = spaces.Box(0, 255, shape=(img_h, img_w, 3), dtype=np.uint8)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)

        self.step_count = 0
        self.spawned = []

    # ---------- Helpers ----------
    def _home(self):
        # 你可以改成自家「安全 home」角度
        self.ctrl.move_arm([0, -1.0, 1.0, 0.0, 1.4, 0.0], duration=1.5)
        self.ctrl.control_gripper(1.0, duration=0.5)

    def _frame(self) -> np.ndarray:
        img = self.camera.get_latest_frame()
        if img is None or img.size == 0:
            img = np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8)
        img = cv2.resize(img, (self.img_w, self.img_h), interpolation=cv2.INTER_AREA)
        return img.astype(np.uint8)

    def _apply(self, dpos, drot, g_open):
        # 以目前 EE 位姿加上增量
        tgt_pos = self.ee_pos + dpos
        # 簡化：保持姿態或少量旋轉（將 RPY 增量累積到 quat）
        rx, ry, rz = drot
        half = np.array([rx/2, ry/2, rz/2], dtype=np.float32)
        dq = np.array([
            math.sin(half[0]), math.sin(half[1]), math.sin(half[2]),
            math.cos(np.linalg.norm(half)+1e-9)  # 粗略；如需嚴謹可用 scipy Rotation
        ], dtype=np.float32)
        tgt_quat = self.ee_quat  # 保持姿態（整理擺放多半不需大旋轉）
        # 解 IK
        code, sol6, _ = solve_ik(tuple(tgt_pos.tolist()), tuple(tgt_quat.tolist()))
        if code >= 0:
            ok = self.ctrl.move_arm(sol6, duration=self.step_dt)
            if ok:
                self.ee_pos = tgt_pos
                self.ee_quat = tgt_quat
        # 夾爪
        self.ctrl.control_gripper(float(np.clip(g_open, 0.0, 1.0)), duration=0.3)

    def _dense_step_reward(self, img: np.ndarray) -> tuple[float, dict]:
        r_align = float(R.reward_alignment(img))
        info = {"align": r_align}
        # 你也可把「分離度/重疊/穩定」做進去；此處先回傳 align
        return r_align, info

    # ---------- Gym API ----------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0

        # 清場 & 生成
        self.spawner.delete_all()
        self.spawned = self.spawner.spawn_random_objects(count=self.objects_n)

        # 回 home
        self._home()

        # 初始化 EE 位姿（從 TF）
        pos, quat = get_current_pose()
        if pos is not None:
            self.ee_pos = np.array(pos, dtype=np.float32)
            self.ee_quat = np.array(quat, dtype=np.float32)

        img = self._frame()
        return img, {"spawned": len(self.spawned)}

    def step(self, action: np.ndarray):
        self.step_count += 1
        a = np.clip(action, -1.0, 1.0)
        dpos = a[:3] * self.pos_scale
        drot = a[3:6] * self.rot_scale
        g_open = (a[6] + 1.0) * 0.5

        self._apply(dpos, drot, g_open)
        time.sleep(self.step_dt)

        img = self._frame()
        r_dense, info = self._dense_step_reward(img)

        terminated = False
        truncated = self.step_count >= self.max_steps

        # 門檻式成功（對齊分數高）
        if r_dense > 0.85:
            terminated = True
            info["success"] = True

        # 回合末再做一次 VLM（昂貴）
        if (terminated or truncated) and self.use_vlm_end:
            r_vlm = float(R.reward_vlm(img, "align objects in a neat row"))
            info["vlm"] = r_vlm
            r_dense = 0.7 * r_dense + 0.3 * r_vlm

        return img, float(r_dense), terminated, truncated, info

    def render(self):
        return self._frame()

    def close(self):
        try:
            self.camera.destroy_node()
            self.ctrl.destroy_node()
            self.spawner.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()
