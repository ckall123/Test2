# xarm6_gym_env.py
import gymnasium as gym
import numpy as np

from camera import TopDownCamera
from gripper_contact import ContactMonitor
from spawner import Spawner
from reward import REWARD_FN
from moveit_controller import MoveItController
import utils
import rclpy


class XArm6Env(gym.Env):
    def __init__(self, cfg: utils.EnvConfig = utils.EnvConfig()):
        if not rclpy.ok():
            rclpy.init(args=None)

        super().__init__()
        self.cfg = cfg
        self.camera = TopDownCamera()
        self.controller = MoveItController()
        self.spawner = Spawner()
        self.contact_monitor = ContactMonitor.get_instance()
        self.reward_fn = REWARD_FN
        self.vlm_clock = utils.VLMClock(cfg.vlm_interval)

        self.joint_dim = 6
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.joint_dim + 1,), dtype=np.float32)
        image_shape = (*cfg.image_size, 3)
        self.observation_space = gym.spaces.Dict({
            'image': gym.spaces.Box(low=0, high=255, shape=image_shape, dtype=np.uint8),
            'proprio': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.joint_dim + 1,), dtype=np.float32),
        })

        self.step_count = 0
        self.attached_obj = None

    def reset(self, *, seed=None, options=None):
        self.step_count = 0
        self.attached_obj = None

        self.spawner.delete_all()
        self.spawner.spawn_random_objects(count=3)

        self.controller.go_home()
        self.controller.move_gripper(0.0)  # ← 修這裡！

        rgb = utils.get_rgb(self.camera, self.cfg)
        proprio = self._get_proprio()
        obs = utils.pack_obs(rgb, proprio)
        return obs


    def step(self, action):
        self.step_count += 1

        q_current = self.controller.get_joint_positions()
        target_joints, gripper_cmd = utils.action_to_targets(action, q_current, self.cfg)
        success, error = self.controller.plan_and_execute(target_joints, gripper_cmd)

        if not success:
            rgb = utils.get_rgb(self.camera, self.cfg)
            obs = utils.pack_obs(rgb, self._get_proprio())
            info = utils.make_info(plan_fail=True)
            info["error"] = error
            return obs, -0.1, True, False, info

        contact_obj = utils.in_contact()
        if contact_obj and self.attached_obj is None:
            if utils.try_attach(contact_obj):
                self.attached_obj = contact_obj
        elif self.attached_obj and gripper_cmd > 0.9:
            utils.try_detach(self.attached_obj)
            self.attached_obj = None

        rgb = utils.get_rgb(self.camera, self.cfg)
        reward, r_info = utils.compute_reward(rgb, self.cfg, self.vlm_clock)
        terminated = utils.check_success(r_info, self.cfg)
        truncated = self.step_count >= self.cfg.max_steps
        obs = utils.pack_obs(rgb, self._get_proprio())
        info = utils.make_info(
            r_align=r_info["r_align"],
            r_vlm=r_info["r_vlm"],
            attached=self.attached_obj,
            step=self.step_count
        )
        return obs, reward, terminated, truncated, info

    def _get_proprio(self):
        joints = self.controller.get_joint_positions()
        gripper = self.controller.get_gripper_state()
        return np.concatenate([joints, [gripper]])

    def render(self, mode='rgb_array'):
        return utils.get_rgb(self.camera, self.cfg)

    def close(self):
        self.camera.destroy_node()
        self.spawner.destroy_node()
        self.controller.shutdown()
        if rclpy.ok():
            rclpy.shutdown()