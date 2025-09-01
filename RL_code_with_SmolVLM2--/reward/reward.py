# =========================
# FILE: reward/reward.py
# =========================
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class XArmReward:
    step_penalty: float = 0.01
    violation_penalty: float = 0.3
    ee_low_penalty: float = 0.1
    move_cost_coeff: float = 0.02
    close_gap_threshold: float = 0.12
    table_z: float = 0.0
    safe_z_margin: float = 0.03
    # 新增：VLM 差分係數
    vlm_delta_coeff: float = 0.5

    def compute(self, obs, action, vlm_delta: float = 0.0) -> float:
        rew = 0.0
        # 時間步開銷
        rew -= float(self.step_penalty)
        # 控制器違規
        if int(obs.get("traj_error", 0)) == 1:
            rew -= float(self.violation_penalty)
        # 末端太低（撞桌風險）
        tcp = np.asarray(obs.get("tcp_pose", np.zeros(7, np.float32)), dtype=np.float32)
        tcp_z = float(tcp[2]) if tcp.shape[0] >= 3 else 10.0
        if tcp_z < (self.table_z + self.safe_z_margin):
            rew -= float(self.ee_low_penalty)
        # 動作成本（L2）
        if action is not None:
            arm = np.asarray(action[:6], dtype=np.float32)
            rew -= float(self.move_cost_coeff * np.linalg.norm(arm, ord=2))
        # 簡單形狀化：夾爪關閉且抬高一點給些微鼓勵
        gpos = float(np.asarray(obs.get("gripper_pos", [1.0]), dtype=np.float32)[0])
        if gpos < self.close_gap_threshold and tcp_z > (self.table_z + 0.15):
            rew += 0.05
        # 重點：把「變得更整齊」的差分獎勵加進來
        rew += float(self.vlm_delta_coeff * float(vlm_delta))
        # 夾住到 [-2, 2]，避免爆掉
        return float(np.clip(rew, -2.0, 2.0))
