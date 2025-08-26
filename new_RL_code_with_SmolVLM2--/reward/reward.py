# =========================
# FILE: reward/reward.py
# =========================
from dataclasses import dataclass
import numpy as np


@dataclass
class XArmReward:
    """將所有獎懲集中在這裡（env 只負責蒐集 obs 與安危判斷）。"""
    table_z: float = 1.0
    safe_z_margin: float = 0.03
    close_gap_threshold: float = 0.015
    move_cost_coeff: float = 0.01
    violation_penalty: float = 1.0
    ee_low_penalty: float = 0.5
    step_penalty: float = 0.01

    def compute(self, obs: dict, action: np.ndarray) -> float:
        rew = 0.0

        # 每步小懲罰，鼓勵快點完成（或避免亂試）
        rew -= self.step_penalty

        # 控制器誤差（Tolerance/Aborted）
        if int(obs.get("traj_error", 0)) == 1:
            rew -= self.violation_penalty

        # 末端太低（可能碰桌）
        tcp_z = float(obs.get("tcp_pose", np.zeros(7))[2])
        if tcp_z < (self.table_z + self.safe_z_margin):
            rew -= self.ee_low_penalty

        # 動作代價（小幅抑制過大關節變化）
        if action is not None:
            # 只看前 6 維（關節），忽略夾爪
            arm = np.asarray(action[:6], dtype=np.float32)
            rew -= float(self.move_cost_coeff * np.linalg.norm(arm, ord=2))

        # 夾爪關閉且末端抬高 => 可能抓起東西，給微小 shaping
        gpos = float(obs.get("gripper_pos", np.zeros(1))[0])
        if gpos < self.close_gap_threshold and tcp_z > (self.table_z + 0.15):
            rew += 0.05

        # 夾住時保持穩定（末端高度穩定）可再加 shaping（可依需求擴充）
        return float(np.clip(rew, -2.0, 1.0))
