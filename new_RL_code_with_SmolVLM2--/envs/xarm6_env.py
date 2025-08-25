# xarm_env.py
import uuid
import time
import random
import gymnasium as gym
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from sensor_msgs.msg import JointState, Image
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory

from gymnasium.spaces import Box, Dict as SpaceDict
from tf2_ros import Buffer, TransformListener

from vlm.sync_api import get_vlm_score
# CHANGED: import TouchCounter and check_done
from reward.reward import compute_rewards, TouchCounter, check_done  # 使用高階 API 一次計算獎勵 + touch 管理

import cv2
from cv_bridge import CvBridge

from rclpy.duration import Duration
from rclpy.time import Time

from objects.sim_object import SimObject
from rl_utils.gripper_control import attach as srv_attach, detach as srv_detach
from objects.spawner import Spawner
from objects.xarm_positions import warm_tf


# -------------------- constants --------------------
# 觀測影像解析度（降低 VLM cost / 記憶體負擔）
IMG_H, IMG_W = 64, 64

# 觀測中保留的物件數量（少量即可訓練 tidy 行為）
N_OBJECTS = 2

# 桌面/工作區界線（給 reward 的參考，用 TABLE_BOUNDS 的 z 來推 table_z）
TABLE_BOUNDS = {
    "x": (-0.75, 0.75),
    "y": (-0.40, 0.40),
    "z": (0.985, 1.015),  # 桌面高度附近（僅參考用）
}

# WORKSPACE 仍保留在檔案中以便某些情況下計分使用，但我們在 compute_rewards 呼叫時會傳入寬鬆 placeholder
WORKSPACE = {"x": (0.30, 0.80), "y": (-0.30, 0.30), "z": (0.00, 0.40)}


class XArm6GymEnv(gym.Env):
    """最小可用的 XArm6 + Gazebo + VLM 獎勵環境。

    - 動作：7 維（6 個關節增量 + 夾爪增量）
    - 觀測：關節/夾爪歸一化狀態 + 下採樣 RGB 影像 + 前 N 個物件位置
    - 獎勵：幾何 + 版面整齊 + VLM 分數；撞桌由 TouchCounter 與 check_done 管理
    """

    # 機械臂各關節（rad）限制，用於 clip
    _LIMITS = np.array([
        [-3.1067,  3.1067],
        [-2.0595,  2.0944],
        [-3.1067,  0.1919],
        [-3.1067,  3.1067],
        [-1.6929,  3.1067],
        [-3.1067,  3.1067],
    ], dtype=np.float32)

    def __init__(self,
                 max_steps=200,
                 vlm_interval=20,
                 vlm_prompt='請以桌面整潔/美觀評分，輸出 0~1 浮點數。',
                 camera_topic='/camera/image_raw',
                 arm_traj_topic='/xarm6_traj_controller/joint_trajectory',
                 grip_action_name='/xarm_gripper_traj_controller/follow_joint_trajectory',
                 gripper_joint_name='drive_joint',
                 # motion knobs
                 arm_step_rad=0.20, arm_limit_margin=0.05, arm_time_sec=0.25,
                 grip_min=0.0, grip_max=0.8552, grip_step=0.08, grip_time_sec=0.25,
                 # robot params
                 robot_model='UF_ROBOT', gripper_link='right_finger',
                 # spawner/arm bounds injection
                 arm_bounds: dict | None = None,
                 object_names: list | None = None,
                 # CHANGED: touch tolerance param (可改)
                 touch_max: int = 5,
                 **kwargs):
        super().__init__()

        # ---- 訓練/回合設定 ----
        self.max_steps = int(max_steps)
        self.vlm_interval = int(vlm_interval)
        self.vlm_prompt = vlm_prompt
        self._last_vlm_score = 0.5  # 啟動前的保守預設
        self.step_count = 0
        self._step_since_vlm = 0

        # ---- ROS 與話題設定 ----
        self.camera_topic = camera_topic
        self.gripper_joint_name = gripper_joint_name
        self.robot_model = robot_model
        self.gripper_link = gripper_link
        self.attached_index = None

        # ---- 動作幅度 ----
        self.arm_step_rad = float(arm_step_rad)
        self.arm_limit_margin = float(arm_limit_margin)
        self.arm_time_sec = float(arm_time_sec)
        self.grip_min = float(grip_min)
        self.grip_max = float(grip_max)
        self.grip_step = float(grip_step)
        self.grip_time_sec = float(grip_time_sec)

        # ---- ROS2 節點與 I/O ----
        node_name = f"xarm6_gym_env_{uuid.uuid4().hex[:6]}"
        self.node: Node = Node(node_name)

        self.arm_pub = self.node.create_publisher(JointTrajectory, arm_traj_topic, 10)
        self.grip_ac = ActionClient(self.node, FollowJointTrajectory, grip_action_name)

        self.state_sub = self.node.create_subscription(JointState, '/joint_states', self._joint_state_cb, 10)
        self.image_sub = self.node.create_subscription(Image, self.camera_topic, self._image_cb, 10)

        # TF：建立後先暖機，讓 TF buffer 有初始資料
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self.node)
        try:
            warm_tf(self.node, self.tf_buffer, spins=40, timeout_sec=0.02)
        except Exception:
            pass

        self.bridge = CvBridge()
        self.latest_image = None

        # ---- 空間定義 ----
        self.action_space = gym.spaces.Box(low=np.array([-1]*7, dtype=np.float32),
                                           high=np.array([1]*7, dtype=np.float32))
        self.observation_space = SpaceDict({
            'state': Box(low=-1.0, high=1.0, shape=(7 + 3*N_OBJECTS,), dtype=np.float32),
            'image': Box(low=0, high=255, shape=(IMG_H, IMG_W, 3), dtype=np.uint8),
        })

        self.current_joint_state = np.zeros(6, dtype=np.float32)
        self.gripper_angle: float = self.grip_max

        # ---- 物件生成器（持久化 Spawner，一次建立，反覆使用）----
        self.arm_bounds = arm_bounds
        self.spawner = Spawner(arm_bounds=self.arm_bounds, object_names=object_names)

        # env 內的 SimObject 列表（env 擁有 SimObject 的生命週期）
        self.objects: list[SimObject] = []

        # CHANGED: TouchCounter 管理碰桌容忍次數
        self.touch_counter = TouchCounter(max_touch=int(touch_max))

        # 內部快取
        self._joint_name_to_idx = None

    # ---------- ROS callbacks ----------
    def _image_cb(self, msg: Image):
        """相機影像 -> OpenCV BGR，失敗則設為 None。"""
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception:
            self.latest_image = None

    def _joint_state_cb(self, msg: JointState):
        """JointState 重排成 joint1~joint6 的順序。"""
        try:
            if self._joint_name_to_idx is None:
                self._joint_name_to_idx = {name: i for i, name in enumerate(msg.name)}
            order = [self._joint_name_to_idx.get(f'joint{i+1}', i) for i in range(6)]
            pos = [msg.position[i] for i in order]
            self.current_joint_state = np.array(pos, dtype=np.float32)
        except Exception:
            self.current_joint_state = np.array(msg.position[:6], dtype=np.float32)

    # ---------- helpers ----------
    def _ee_pos(self, joints: np.ndarray) -> np.ndarray:
        """取末端 TCP 的世界座標；若 TF 不可得，使用簡易幾何近似。"""
        try:
            tf = self.tf_buffer.lookup_transform('world', 'link_tcp', Time(), timeout=Duration(seconds=0.05))
            p = tf.transform.translation
            return np.array([p.x, p.y, p.z], dtype=np.float32)
        except Exception:
            # 極簡近似（保證不中斷訓練）
            x = 0.3 + 0.2*np.cos(joints[0]) + 0.2*np.cos(joints[0] + joints[1])
            y = 0.0 + 0.2*np.sin(joints[0]) + 0.2*np.sin(joints[0] + joints[1])
            z = 0.05 + 0.1*(joints[2] + 1.5)
            return np.array([x, y, max(0.0, z)], dtype=np.float32)

    def _normalize_obs(self, raw_state: np.ndarray) -> np.ndarray:
        """將關節與夾爪映射到 [-1,1]，其餘直接接上。"""
        joint_scaled = 2*(self.current_joint_state - self._LIMITS[:,0])/(self._LIMITS[:,1]-self._LIMITS[:,0]) - 1
        grip_scaled  = 2*(self.gripper_angle - self.grip_min)/(self.grip_max - self.grip_min) - 1
        return np.concatenate([joint_scaled, [grip_scaled], raw_state[7:]]).astype(np.float32)

    def _layout_score(self) -> float:
        """薄包一層方便呼叫（保留給需要時直接查 layout）。"""
        P = [obj.position for obj in self.objects]
        # 若需要單獨 layout，可自行在這裡引入 reward.layout_score
        try:
            from reward.reward import layout_score
            return layout_score(P)
        except Exception:
            return 0.0

    def _send_gripper_goal(self, pos: float):
        """非阻塞地送出夾爪 FollowJointTrajectory 目標。"""
        pos = float(np.clip(pos, self.grip_min, self.grip_max))
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = [self.gripper_joint_name]
        pt = JointTrajectoryPoint()
        pt.positions = [pos]
        sec = int(self.grip_time_sec); nsec = int((self.grip_time_sec - sec) * 1e9)
        pt.time_from_start.sec = sec; pt.time_from_start.nanosec = nsec
        goal.trajectory.points = [pt]
        try:
            if not self.grip_ac.server_is_ready():
                self.grip_ac.wait_for_server(timeout_sec=2.0)
            self.grip_ac.send_goal_async(goal)
        except Exception:
            pass

    # ---------- object spawn helper wrapper ----------
    def random_spawn_objects(self, num_objects: int) -> list:
        """呼叫 Spawner.spawn_random_objects 並建立 SimObject list（回傳成功建立的 SimObject）。"""
        spawned = self.spawner.spawn_random_objects(num_objects)
        sim_objs = []
        for name, pos in spawned:
            sim_obj = SimObject(name=name, position=pos)
            sim_objs.append(sim_obj)
        self.objects.extend(sim_objs)
        return sim_objs

    # ---------- Gym API ----------
    def step(self, action: np.ndarray):
        # 1) 發送關節/夾爪命令
        self.step_count += 1
        a = np.clip(action, -1.0, 1.0)
        target = self.current_joint_state + self.arm_step_rad * a[:6]
        lo = self._LIMITS[:,0] + self.arm_limit_margin
        hi = self._LIMITS[:,1] - self.arm_limit_margin
        target = np.clip(target, lo, hi)

        arm = JointTrajectory(); arm.joint_names = [f'joint{i+1}' for i in range(6)]
        pt = JointTrajectoryPoint(); pt.positions = target.tolist()
        sec = int(self.arm_time_sec); nsec = int((self.arm_time_sec - sec) * 1e9)
        pt.time_from_start.sec = sec; pt.time_from_start.nanosec = nsec
        arm.points.append(pt); self.arm_pub.publish(arm)

        self.gripper_angle = float(np.clip(self.gripper_angle + self.grip_step * a[6], self.grip_min, self.grip_max))
        self._send_gripper_goal(self.gripper_angle)

        # 2) 拉一次 ROS event queue
        rclpy.spin_once(self.node, timeout_sec=0.0)

        # 3) 夾取/放置邏輯（簡單版）
        ee = self._ee_pos(self.current_joint_state)
        if len(self.objects) > 0:
            dists = np.linalg.norm(np.stack([o.position for o in self.objects]) - ee, axis=1)
            near_idx = int(np.argmin(dists))
            near_dist = float(dists[near_idx])
        else:
            near_idx, near_dist = -1, float('inf')

        close_threshold = 0.08
        grab_threshold = self.grip_min + 0.7*(self.grip_max - self.grip_min)
        release_threshold = self.grip_min + 0.3*(self.grip_max - self.grip_min)

        if (self.gripper_angle > grab_threshold) and (near_dist < close_threshold):
            if self.attached_index is None and near_idx >= 0:
                try:
                    srv_attach(self.robot_model, self.gripper_link, self.objects[near_idx].name, self.objects[near_idx].link)
                    self.attached_index = near_idx
                    self.objects[near_idx].attached = True
                except Exception:
                    pass
            elif self.attached_index is not None:
                idx = self.attached_index
                self.objects[idx].position = np.array([ee[0], ee[1], 0.05], np.float32)

        if (self.gripper_angle < release_threshold) and (self.attached_index is not None):
            idx = self.attached_index
            try:
                srv_detach(self.robot_model, self.gripper_link, self.objects[idx].name, self.objects[idx].link)
            except Exception:
                pass
            self.attached_index = None
            self.objects[idx].attached = False
            # 放置：就地落下（不做 Y 對齊作弊）
            self.objects[idx].position = np.array([ee[0], ee[1], 0.05], np.float32)

        # 4) 觀測
        obs = self._obs()

        # 5) 週期性 VLM 評分
        self._step_since_vlm += 1
        if ((self.vlm_interval == 0) or (self._step_since_vlm >= self.vlm_interval)) and (self.latest_image is not None):
            try:
                resized = cv2.resize(self.latest_image, (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR)
                score = float(get_vlm_score(resized, prompt=self.vlm_prompt))
                self._last_vlm_score = max(0.0, min(1.0, score))
                self._step_since_vlm = 0
            except Exception:
                pass

        # 6) 使用高階 API 計算獎勵（幾何 + 版面 + VLM + 觸桌懲罰）
        # CHANGED: we pass a very loose workspace placeholder instead of restricting X/Y
        big_workspace = {'x': (-1e6, 1e6), 'y': (-1e6, 1e6), 'z': (-1e6, 1e6)}
        rewards = compute_rewards(
            ee=ee,
            objects=[o.position for o in self.objects],
            tf_buffer=self.tf_buffer,
            vlm_score=self._last_vlm_score,
            workspace=big_workspace,  # loose placeholder; actual done uses z & touch_counter
        )

        # CHANGED: update touch counter based on detected touched flag
        self.touch_counter.update(bool(rewards.touched))

        # 7) 終止條件（terminated: object fell or repeated table touch; truncated: timeout）
        # CHANGED: use check_done to get terminated & reason
        table_z = float(np.mean(TABLE_BOUNDS['z']))  # use TABLE_BOUNDS z as table reference
        terminated, done_info = check_done(
            objects=[o.position for o in self.objects],
            ee=ee,
            touch_counter=self.touch_counter,
            table_z=table_z,
            drop_tol=0.05,
        )

        truncated = bool(self.step_count >= self.max_steps)

        # prepare info
        info = {
            'gripper_position': self.gripper_angle,
            'vlm_score': self._last_vlm_score,
            'touched_table': bool(rewards.touched),
            'touch_count': int(self.touch_counter.count),
            'rewards': rewards.to_dict() if hasattr(rewards, 'to_dict') else rewards.__dict__,
        }
        if done_info:
            info.update(done_info)

        # debug print when an episode ends quickly (optional)
        if terminated or truncated:
            print("=== EPISODE END ===")
            print("step_count:", self.step_count, "terminated:", terminated, "truncated:", truncated)
            print("touch_count:", self.touch_counter.count, "touched_flag:", rewards.touched)
            print("object_positions:", [o.position.tolist() for o in self.objects])
            print("ee:", ee.tolist())
            print("done_info:", done_info)

        return obs, float(rewards.final), terminated, truncated, info

    def _obs(self):
        raw = list(self.current_joint_state) + [self.gripper_angle]
        for i in range(N_OBJECTS):
            if i < len(self.objects):
                raw.extend(self.objects[i].position.tolist())
            else:
                raw.extend([0.0, 0.0, 0.0])
        img = np.zeros((IMG_H, IMG_W, 3), np.uint8) if self.latest_image is None else cv2.resize(self.latest_image, (IMG_W, IMG_H))
        return {'state': self._normalize_obs(np.asarray(raw, np.float32)), 'image': img}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self._step_since_vlm = 0
        self.current_joint_state[:] = 0.0
        self.gripper_angle = self.grip_max
        self._send_gripper_goal(self.gripper_angle)
        self.attached_index = None

        # CHANGED: reset touch counter on new episode
        self.touch_counter = TouchCounter(max_touch=self.touch_counter.max_touch)

        # 清除上回合自己 spawn 的物件（持久 Spawner，乾淨好控）
        try:
            self.spawner.delete_all()
        except Exception:
            pass
        self.objects = []
        time.sleep(0.10)

        # 重新 spawn 最多 N_OBJECTS 個物件（簡單呼叫 wrapper）
        self.random_spawn_objects(N_OBJECTS)

        print('-------------------[RESET] env reset called-----------------------')
        return self._obs(), {}

    def close(self):
        try:
            if hasattr(self, 'spawner') and self.spawner is not None:
                self.spawner.destroy_node()
        except Exception:
            pass
        try:
            if hasattr(self, 'node'):
                self.node.destroy_node()
        except Exception:
            pass
