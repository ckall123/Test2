from torch.utils.tensorboard import SummaryWriter

class TensorboardCallback:
    """
    訓練過程中記錄 reward 與 episode 資訊到 TensorBoard。
    可接入任何訓練迴圈中，定期呼叫 log_episode()。
    """
    def __init__(self, log_dir="runs/xarm6"):
        self.writer = SummaryWriter(log_dir)
        self.episode_count = 0

    def log_episode(self, reward: float, length: int):
        """將一集的 reward 與長度寫入 TensorBoard"""
        self.writer.add_scalar("Episode/Reward", reward, self.episode_count)
        self.writer.add_scalar("Episode/Length", length, self.episode_count)
        self.episode_count += 1

    def close(self):
        self.writer.close()