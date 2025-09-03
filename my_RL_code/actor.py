import torch
from lerobot import SmolVLA
import numpy as np


class SmolVLAActor:
    def __init__(self, model_path="lerobot/smolvla_base", device="cuda"):
        self.device = torch.device(device)
        self.model = SmolVLA.from_pretrained(model_path).to(self.device)
        self.model.eval()

        # ✅ 可調參數
        self.instruction = "align objects in a row"  # 預設指令，可改為其他任務
        self.clip_range = 0.05  # 限制動作範圍，防止暴走

    def act(self, image: np.ndarray, agent_pos: np.ndarray) -> np.ndarray:
        """
        輸入：
          - image: HxWx3 RGB np.uint8 (例如 64x64)
          - agent_pos: np.array [x,y,z,gripper]
        回傳：
          - action: np.array [dx, dy, dz, gripper]
        """
        # ✅ 視覺處理
        image = torch.from_numpy(image / 255.0).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        state = torch.from_numpy(agent_pos).unsqueeze(0).float().to(self.device)

        # ✅ 推論（含任務指令）
        output = self.model.infer({
            "vision": image,
            "state": state,
            "text": self.instruction
        })

        # ✅ 後處理：裁剪避免超出範圍
        action = output.squeeze(0).cpu().numpy()
        action[:3] = np.clip(action[:3], -self.clip_range, self.clip_range)
        action[3] = float(np.clip(action[3], 0.0, 1.0))

        return action


# ✅ 整合環境使用接口（選用）
def get_action_from_actor(image: np.ndarray, agent_pos: np.ndarray) -> np.ndarray:
    if not hasattr(get_action_from_actor, "actor"):
        get_action_from_actor.actor = SmolVLAActor()
    return get_action_from_actor.actor.act(image, agent_pos)
