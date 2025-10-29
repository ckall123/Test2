#!/usr/bin/env python3
"""
xarm6_gym_env.py

實驗流程（高可視度註解版）：
- reset():
    1) 「soft reset」：不清場，只輪換本輪的 current_target（每回合只處理一個物體）
    2) 控制器回到 home、打開夾爪（避免殘留力/接觸）
    3) 量測當前場景 → 初始化三個基準：
       - prev_E：幾何能量（全局整齊度，越小越整齊）
       - prev_dist：TCP→目標物距離（proximity 成形的比較基準）
       - VLMThrottle.reset(init_score)：紀錄一張初始影像的 VLM 分數
    4) 組裝觀測（image/state）並回傳 info（含 target 與 E）

- step(action):
    1) 作用動作（6關節增量 + 夾爪開合），記錄到 ActionHistory
    2) 重新量測場景 → 得到：
       - E_now：目前幾何能量；ΔE = prev_E - E_now（主獎勵）
       - d_now：目前距離；Δd = prev_dist - d_now（proximity 成形）
    3) 受控 VLM 加分：只有在 ΔE>門檻 且 達到步距間隔 才會打分，回傳分數差（夾限）
    4) 距離里程碑：基於 d_now 的分段獎勵（5/4/3/2/1 cm）
    5) 動作成本：L2 正則抑制大動作
    6) 總獎勵：ΔE + prox + vlm_bonus + dist_bonus - action_cost
    7) 成功/截斷、更新基準（prev_E=E_now, prev_dist=d_now），回傳觀測與 info（含分項貢獻）
"""
import time
from typing import Dict, Tuple, Any, List
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from camera import TopDownCamera
from moveit_controller import MoveItController
from spawner import Spawner
from gripper_contact import ContactMonitor

from pose_tracker import PoseTracker, load_target_names
from reward import compute_geometry_energy
from vlm import VLMScorer
import utils


class XArm6Env(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, node, executor, cfg: utils.EnvConfig = utils.EnvConfig()):
        super().__init__()
        self.node = node
        self.executor = executor
        self.cfg = cfg

        # 控制/感測
        self.controller = MoveItController(node, executor)
        self.camera = TopDownCamera(node)
        self.spawner = Spawner(node, executor)
        self.contact_monitor = ContactMonitor(node)
        self.tracker = PoseTracker(node)
        self.vlm = VLMScorer()
        self.vlm_throttle = utils.VLMThrottle(cfg.vlm_interval, cfg.vlm_deltaE_thresh, cfg.vlm_clip)

        # 目標物（soft reset：只輪換，不清場）
        self.object_names: List[str] = load_target_names()
        assert len(self.object_names) >= 1, "需要至少 1 個目標物"
        self.round_idx: int = 0
        self.current_target: str = self.object_names[self.round_idx]

        # 動作/觀測空間
        self.joint_dim = 6
        self.act_dim = self.joint_dim + 1  # + gripper
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.act_dim,), dtype=np.float32)

        w, h = self.cfg.image_size  # (W,H)
        image_shape = (h, w, 3)     # HWC
        self.hist_dim = self.cfg.action_hist_len * self.act_dim
        self.state_dim = (self.joint_dim + 1) + 3 + 1 + 1 + self.hist_dim
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=255, shape=image_shape, dtype=np.uint8),
            "state": spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32),
        })

        # 狀態
        self.step_count: int = 0
        self.prev_E: float = 0.0
        self.prev_dist: float = np.inf
        self.action_hist = utils.ActionHistory(self.cfg.action_hist_len, action_dim=self.act_dim)

        # 幾何能量的桌面邊界（依場景調整）
        self.table_bounds: Tuple[Tuple[float, float], Tuple[float, float]] = ((-0.30, 0.30), (-0.20, 0.20))

    # ============== Gym API ==============
    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        """
        實驗「一回合開始」：
        - soft reset：不清場、只換 current_target（round-robin）
        - 控制器回 home / 打開夾爪
        - 初始化 E/距離/VLM 分數基準
        """
        super().reset(seed=seed)
        self.step_count = 0
        self.action_hist.clear()

        self.node.get_logger().info("[reset] Begin reset")

        self.current_target = self.object_names[self.round_idx % len(self.object_names)]
        self.round_idx = (self.round_idx + 1) % len(self.object_names)

        self.node.get_logger().info("[reset] Before go_home")
        self.controller.go_home()
        self.controller.move_gripper(1.0)
        self.node.get_logger().info("[reset] After move_gripper")

        start = time.perf_counter()
        objs = self._gather_objects_state()
        self.node.get_logger().info(f"[reset] After gather_objects, {len(objs)} objects, took {time.perf_counter() - start:.2f}s")

        self.prev_E = compute_geometry_energy(objs, self.table_bounds, self.cfg)
        self.prev_dist = self._distance_to_target()
        self.vlm_throttle.reset(0.0)
        obs = self._get_obs()
        self.node.get_logger().info("[reset] End reset")
        info = {"target": self.current_target, "E": self.prev_E}
        return obs, info

    def step(self, action: np.ndarray):
        """
        實驗「單步交互」：
        - 執行動作 → 量測（E_now, d_now）
        - 計算獎勵分項（ΔE / Δd / VLM / 里程碑 / 成本）
        - 成功或截斷 → 更新基準 → 回傳觀測與 info（高可視度分項）
        """
        self.step_count += 1

        # (1) 作用動作（6關節增量 + 夾爪）並記錄歷史
        self._apply_action(action)
        self.action_hist.add(action)

        # (2) 量測當前場景：幾何能量與距離
        objs = self._gather_objects_state()
        E_now = compute_geometry_energy(objs, self.table_bounds, self.cfg)
        delta_E = float(self.prev_E - E_now)

        d_now = self._distance_to_target()
        delta_d = float(self.prev_dist - d_now)

        # (3) 獎勵分項
        prox = self.cfg.prox_weight * delta_d  # 距離縮短成形
        vlm_bonus = self.vlm_throttle.maybe_bonus(delta_E, self._grab_camera_rgb(), self.vlm)  # 受控 VLM
        dist_bonus = utils.distance_sparse_bonus(d_now, self.cfg.distance_bins, self.cfg.distance_vals)  # 里程碑
        action_cost = utils.compute_action_cost(action, self.cfg.action_cost_coef)  # 動作成本

        reward = delta_E + prox + vlm_bonus + dist_bonus - action_cost

        # (4) 成功/截斷條件
        terminated = bool(E_now <= self.cfg.E_success)
        truncated = bool(self.step_count >= self.cfg.max_steps)

        # (5) 更新基準供下一步比較
        self.prev_E = E_now
        self.prev_dist = d_now

        # (6) 回傳觀測與高可視度 info（便於你在訓練時對照調參）
        obs = self._get_obs()
        info = {
            "step": self.step_count,
            "target": self.current_target,
            # 全局整齊度
            "E": E_now,
            "delta_E": delta_E,
            # 距離相關
            "distance": d_now,
            "delta_d": delta_d,            # >0 表示更接近目標
            "cos_align": self._tcp_heading_cos_to_target(),
            # 獎勵分項
            "prox_contrib": prox,
            "vlm_bonus": vlm_bonus,
            "dist_bonus": dist_bonus,
            "action_cost": action_cost,
        }
        return obs, reward, terminated, truncated, info

    def render(self) -> np.ndarray:
        return self._capture_rgb()

    def close(self):
        pass  # 使用 soft reset，不在此清場

    # ============== Helpers ==============
    def _apply_action(self, action: np.ndarray) -> None:
        """連續動作 ∈ [-1,1]^(6+1) → 6關節增量與夾爪命令，交給 MoveIt 執行。"""
        a = np.asarray(action, dtype=np.float32)

        # 1) 目標關節 & 夾爪
        dq = a[:self.joint_dim] * self.cfg.action_scale
        q_now = self.controller.get_joint_positions()
        q_tar = q_now + dq
        g_cmd = float(np.clip((a[-1] + 1.0) * 0.5, 0.0, 1.0))

        # 2) 執行規劃（抓成功/錯誤）
        try:
            result = self.controller.plan_and_execute(q_tar, g_cmd)
            if isinstance(result, tuple) and len(result) >= 1:
                success = bool(result[0]); err = result[1] if len(result) > 1 else ""
            else:
                success, err = True, ""
        except Exception as e:
            success, err = False, str(e)

        # 3) ★ 關鍵：pump executor，讓 action 回調與 Gazebo/TF 服務得以處理
        try:
            for _ in range(8):  # 約 80ms
                self.executor.spin_once(timeout_sec=0.01)
        except Exception:
            pass

        # 4) 每 10 步打印一次，確認真的在下動作
        if self.step_count % 10 == 1:
            self.node.get_logger().info(
                f"[apply] step={self.step_count} |dq|={np.linalg.norm(dq):.4f}, g={g_cmd:.2f}, ok={success}, err={err}"
            )


    def _grab_camera_rgb(self) -> np.ndarray:
        """
        透過 TopDownCamera 取得最新影像（RGB，HxWx3）。
        若暫時取不到，回傳黑圖（大小用 cfg.image_size 決定）。
        """
        img = self.camera.get_latest_frame(self.executor)
        if img is None or not isinstance(img, np.ndarray) or img.ndim != 3 or img.shape[2] != 3:
            w, h = self.cfg.image_size  # (W,H)
            return np.zeros((h, w, 3), dtype=np.uint8)
        return img

    def _capture_rgb(self) -> np.ndarray:
        """取 top-down 影像並縮放到 EnvConfig.image_size（HWC, uint8）。"""
        rgb = self._grab_camera_rgb()
        return utils.preprocess_image(rgb, self.cfg.image_size)

    def _get_obs(self) -> Dict[str, Any]:
        """組裝觀測：image（HWC）與 state（連續特徵向量）。"""
        image = self._capture_rgb()
        state = self._build_state_features()
        return {"image": image, "state": state.astype(np.float32)}

    def _build_state_features(self) -> np.ndarray:
        """state 向量：joints(6)+gripper(1)+rel(3)+dist(1)+cos(1)+action_hist(N×7)。"""
        q = self.controller.get_joint_positions().astype(np.float32)
        g = np.array([self.controller.get_gripper_state()], dtype=np.float32)
        rel = self._rel_vec_to_target()
        d = np.array([np.linalg.norm(rel)], dtype=np.float32)
        cos_align = np.array([self._tcp_heading_cos_to_target()], dtype=np.float32)
        hist_vec = self.action_hist.vector()
        if hist_vec.size == 0:
            hist_vec = np.zeros(self.hist_dim, dtype=np.float32)
        parts = [q, g, rel.astype(np.float32), d, cos_align, hist_vec.astype(np.float32)]
        return np.concatenate(parts, axis=0)

    def _gather_objects_state(self) -> List[Dict[str, Any]]:
        """取所有物體的 pos/yaw/radius，供 compute_geometry_energy(E) 使用。"""
        return self.tracker.get_object_states(self.object_names, radius_lookup=None)

    def _rel_vec_to_target(self) -> np.ndarray:
        """目標物相對 TCP 的向量；取不到時回 0 向量，以免觀測 NaN。"""
        rel = self.tracker.rel_vec_to(self.current_target)
        if not np.isfinite(rel).all():
            return np.zeros(3, dtype=np.float32)
        return rel
    
    def _distance_to_target(self) -> float:
        """TCP→目標物的歐氏距離（公尺）。"""
        return float(np.linalg.norm(self._rel_vec_to_target()))

    def _tcp_heading_cos_to_target(self) -> float:
        """TCP 航向與目標方向（XY）的夾角餘弦，越接近 1 越對準。"""
        tcp = self.tracker.get_tcp_pose()
        if tcp is None:
            return 0.0
        tcp_pos, tcp_yaw = tcp

        tgt_pos = self.tracker.get_object_pose_world(self.current_target)
        if tgt_pos is None:
            return 0.0

        v = tgt_pos[:2] - tcp_pos[:2]
        nv = v / (np.linalg.norm(v) + 1e-6)
        heading = np.array([np.cos(tcp_yaw), np.sin(tcp_yaw)], dtype=np.float32)
        return float(np.clip(np.dot(heading, nv), -1.0, 1.0))

    def _score_vlm_rgb(self) -> float:
        """以當前 top-down 影像向 VLM 評分（0~1），只在 reset 用於初始化節流器。"""
        img_rgb = self._grab_camera_rgb()
        return float(self.vlm.score_image(img_rgb, instruction="align objects in a row"))
