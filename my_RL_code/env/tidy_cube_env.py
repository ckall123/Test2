import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import rclpy

from env.camera import TopDownCamera
from env.fk import FKClient
from my_RL_code.env.ik import IKClient
from env.gripper import GripperStateTracker
from env.joint_controller import JointController
from object.spawner import Spawner
from env import reward

JOINT_ORDER = ['joint1','joint2','joint3','joint4','joint5','joint6']


def _ordered_arm_positions(joint_msg):
    """Reorder /joint_states into JOINT_ORDER. Falls back to zeros on mismatch."""
    name_to_pos = dict(zip(joint_msg.name, joint_msg.position))
    try:
        return np.array([name_to_pos[n] for n in JOINT_ORDER], dtype=np.float32)
    except Exception:
        return np.zeros(6, dtype=np.float32)


def _opencv_has_gui() -> bool:
    """Detect whether OpenCV has a GUI backend (GTK/Qt/Win/Cocoa)."""
    try:
        info = cv2.getBuildInformation().lower()
        return any(k in info for k in ["gtk", "qt", "win32", "cocoa"])
    except Exception:
        return False


class TidyCubeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, obs_type="pixels_agent_pos", resolution=(64, 64)):
        super().__init__()
        self.obs_type = obs_type
        self.render_mode = render_mode
        # resolution as (W, H)
        self.resolution = resolution
        self._cv2_has_gui = _opencv_has_gui()

        if not rclpy.ok():
            rclpy.init()

        # ROS2 nodes/clients
        self.camera = TopDownCamera(resolution=self.resolution)
        self.fk = FKClient()
        self.ik = IKClient()
        self.gripper = GripperStateTracker()
        self.controller = JointController()
        self.spawner = Spawner()

        # Action: [dx, dy, dz, gripper]
        self.action_space = spaces.Box(
            low=np.array([-0.05, -0.05, -0.05, 0.0], dtype=np.float32),
            high=np.array([ 0.05,  0.05,  0.05, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Observation: Dict with image (HWC uint8) and agent_pos (ee_x, ee_y, ee_z in meters, gripper in [0,1])
        W, H = self.resolution
        image_shape = (H, W, 3)
        self.observation_space = spaces.Dict({
            "pixels": spaces.Box(low=0, high=255, shape=image_shape, dtype=np.uint8),
            "agent_pos": spaces.Box(
                low=np.array([-3.0, -3.0, -3.0, 0.0], dtype=np.float32),
                high=np.array([ 3.0,  3.0,  3.0, 1.0], dtype=np.float32),
                dtype=np.float32,
            ),
        })

    # ---------------------------- helpers ----------------------------
    def _get_frame(self):
        """Get an RGB frame resized to observation space; fallback to black if None."""
        img = self.camera.get_latest_frame()
        if img is None:
            # Nudge the ROS callbacks once to try to receive a frame
            try:
                rclpy.spin_once(self.camera, timeout_sec=0.05)
            except Exception:
                pass
            img = self.camera.get_latest_frame()

        W, H = self.resolution
        if img is None:
            return np.zeros((H, W, 3), dtype=np.uint8)
        if img.shape[:2] != (H, W):
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        if img.dtype != np.uint8:
            img = img.astype(np.uint8, copy=False)
        return np.ascontiguousarray(img)

    def _get_obs(self):
        img = self._get_frame()

        joint_msg = getattr(self.gripper, 'last_msg', None)
        if joint_msg is None:
            try:
                rclpy.spin_once(self.gripper, timeout_sec=0.05)
            except Exception:
                pass
            joint_msg = getattr(self.gripper, 'last_msg', None)

        arm_q = _ordered_arm_positions(joint_msg) if joint_msg else np.zeros(6, np.float32)
        ee_pos = self.fk.compute_fk(JOINT_ORDER, arm_q.tolist())
        if ee_pos is None or (isinstance(ee_pos, (list, tuple, np.ndarray)) and not np.any(ee_pos)):
            ee_pos = np.zeros(3, dtype=np.float32)
        else:
            ee_pos = np.asarray(ee_pos, dtype=np.float32)

        grip = float(self.gripper.get_state())
        agent_pos = np.concatenate([ee_pos, [grip]]).astype(np.float32)
        return {"pixels": img, "agent_pos": agent_pos}

    # ---------------------------- gymnasium API ----------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Move to a neutral/home pose
        home = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        try:
            self.controller.send_joint_positions(home, duration=1.5)
        except Exception:
            pass

        # Re-spawn scene objects
        try:
            self.spawner.delete_all()
            self.spawner.spawn_random_objects(count=3)
        except Exception:
            pass

        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        # Clip action to valid bounds
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        cur = self._get_obs()["agent_pos"][:3]
        target = (cur + action[:3]).tolist()

        q = self.ik.compute_ik(target)
        if q is None or len(q) != 6:
            obs = self._get_obs()
            return obs, -1.0, False, False, {"ik_failed": True}

        try:
            self.controller.send_joint_positions(q, duration=0.6)
            self.controller.control_gripper(float(action[3]), duration=0.4)
        except Exception:
            pass

        obs = self._get_obs()
        rew = float(reward.reward_horizontal_alignment(obs["pixels"]))
        terminated = False
        truncated = False
        return obs, rew, terminated, truncated, {}

    def render(self):
        if self.render_mode == "human":
            if not self._cv2_has_gui:
                return  # headless: no-op
            cv2.imshow("TidyCubeEnv", self._get_frame())
            cv2.waitKey(int(1000 / self.metadata["render_fps"]))
        elif self.render_mode == "rgb_array":
            return self._get_frame()

    def close(self):
        # Only try to close windows if GUI is present
        if getattr(self, "_cv2_has_gui", False):
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
        try:
            self.camera.destroy_node()
            self.gripper.destroy_node()
            self.fk.destroy_node()
            self.ik.destroy_node()
            self.controller.destroy_node()
            self.spawner.destroy_node()
        finally:
            if rclpy.ok():
                try:
                    rclpy.shutdown()
                except Exception:
                    pass
