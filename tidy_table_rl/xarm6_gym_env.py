import gymnasium as gym
import numpy as np
import cv2
import time

from gymnasium import spaces
from moveit_controller import MoveIt2Controller
from spawner import Spawner
from collision_object import add_table
from camera import capture_rgb
from reward import reward_alignment, reward_vlm, reward_combined


class XArm6TidyEnv(gym.Env):
    def __init__(self,
                 image_size=(96, 96),
                 max_steps=50,
                 end_only_vlm=True,
                 vlm_every_n=10,
                 reward_weights=(0.7, 0.3),
                 object_count=3,
                 seed=None):
        super().__init__()

        self.image_size = image_size
        self.max_steps = max_steps
        self.end_only_vlm = end_only_vlm
        self.vlm_every_n = vlm_every_n
        self.w_align, self.w_vlm = reward_weights
        self.object_count = object_count

        self.action_space = spaces.Box(low=np.array([-0.05]*6 + [0.0]),
                                       high=np.array([0.05]*6 + [1.0]),
                                       dtype=np.float32)

        obs_shape = (image_size[1], image_size[0], 3)
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=obs_shape, dtype=np.uint8)

        self.controller = MoveIt2Controller()
        self.spawner = Spawner()

        self.rng = np.random.default_rng(seed)
        self._step_count = 0
        self._cached_vlm_scores = {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        self._cached_vlm_scores.clear()

        # 清空場景
        self.spawner.clear_all()
        add_table()

        # 生成新物件
        self.object_infos = self.spawner.spawn_objects(self.object_count)

        # 回 home + 張開爪
        self.controller.move_arm_home()
        self.controller.control_gripper(1.0)

        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        delta_joints = action[:6]
        gripper_cmd = action[6]

        done = False
        truncated = False
        info = {}

        # 控制手臂
        ok1 = self.controller.move_arm([a + d for a, d in zip(self.controller.get_current_joint_values(), delta_joints)])
        ok2 = self.controller.control_gripper(gripper_cmd)
        if not ok1 or not ok2:
            done = True
            reward = -1.0
            return self._get_obs(), reward, done, truncated, info

        img = self._get_obs()

        align_score = reward_alignment(img)
        vlm_score = 0.0

        use_vlm = not self.end_only_vlm and (self._step_count % self.vlm_every_n == 0)
        is_final_step = self._step_count + 1 >= self.max_steps

        if use_vlm or (self.end_only_vlm and is_final_step):
            key = self._hash_img(img)
            if key in self._cached_vlm_scores:
                vlm_score = self._cached_vlm_scores[key]
            else:
                vlm_score = reward_vlm(img)
                self._cached_vlm_scores[key] = vlm_score

        reward = reward_combined(align_score, vlm_score, self.w_align, self.w_vlm)

        self._step_count += 1
        done = self._step_count >= self.max_steps

        info.update({
            'align_score': align_score,
            'vlm_score': vlm_score,
            'held_object': self.controller.held_object,
            'step': self._step_count
        })

        return img, reward, done, truncated, info

    def _get_obs(self):
        img = capture_rgb(size=self.image_size)
        return img

    def _hash_img(self, img):
        return hash(img.tobytes())

    def close(self):
        self.controller.destroy_node()

    def render(self):
        img = self._get_obs()
        cv2.imshow("Top-Down View", img)
        cv2.waitKey(1)

    def seed(self, seed=None):
        self.rng = np.random.default_rng(seed)
        return [seed]