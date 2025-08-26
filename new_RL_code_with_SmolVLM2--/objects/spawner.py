# object/spawner.py
import uuid
import random
import numpy as np
import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SpawnEntity, DeleteEntity
from typing import List, Tuple, Optional

# -------------------------------------------------
# 桌面生成範圍（與 TABLE_BOUNDS 對齊）
# -------------------------------------------------
# X_RANGE = [-0.75, 0.75]
# Y_RANGE = [-0.40, 0.40]

X_RANGE = [-1.05, 0.45]
Y_RANGE = [-1.20, -0.40]
Z_HEIGHT = 1.015  # 物件 z 放在桌面高度，按需微調


# 手臂佔用區域（避免在此生成）
# Arm_X_RANGE = (-0.05, 0.257)
# Arm_Y_RANGE = (-0.116, 0.116)
# Arm_Z_RANGE = (-0.05, 0.601)

Arm_X_RANGE = (-0.45, -0.11)
Arm_Y_RANGE = (-0.67, -0.26)
Arm_Z_RANGE = (1.01, 1.616)

# 可供生成的模型名稱（預設改為空，請在建構時或呼叫時提供）
DEFAULT_OBJECT_NAMES = ["beer"]


class Spawner(Node):
    """Gazebo Entity Spawner（隨機版）

    只提供隨機生成 API：`spawn_random_objects`。
    - 不再公開 `spawn_object`。
    - 你只需要提供：
        1) 要生成的數量 `count`
        2) 可被隨機挑選的模型清單 `candidates`（可選；若不給就用建構子傳入的 `object_names`）
      並可選擇啟用 `strict_unique=True` 以保證「同一個 base model 不重複」。
    - 自動避開手臂 XY 投影區域，並盡量避免物件彼此太近。
    """

    def __init__(self,
                 *,
                 arm_bounds: dict | None = None,
                 object_names: Optional[List[str]] = None,
                 node_name: Optional[str] = None):
        node_name = node_name or f"spawner_node_{uuid.uuid4().hex[:8]}"
        super().__init__(node_name)

        self.spawn_cli = self.create_client(SpawnEntity, '/spawn_entity')
        self.delete_cli = self.create_client(DeleteEntity, '/delete_entity')

        self.arm_bounds = arm_bounds or {
            'x': Arm_X_RANGE,
            'y': Arm_Y_RANGE,
            'z': Arm_Z_RANGE,
        }

        # 使用者自己提供的 object 名單；若未提供則為空清單
        self.object_names = list(object_names) if object_names else list(DEFAULT_OBJECT_NAMES)
        if not self.object_names:
            self.get_logger().warn("No object_names provided; you must pass candidates to spawn_random_objects().")

        # 記錄目前 spawn 的 instance name（用來 delete_all）
        self.spawned_names: List[str] = []

        # 服務等待（短暫重試）
        for _ in range(5):
            if self.spawn_cli.wait_for_service(timeout_sec=1.0) and \
               self.delete_cli.wait_for_service(timeout_sec=1.0):
                break

    # ---------- 刪除相關 ----------
    def delete_object(self, name: str, timeout: float = 8.0) -> bool:
        req = DeleteEntity.Request()
        req.name = name
        future = self.delete_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        # 若 delete 成功或已刪除，移出 spawned_names
        if name in self.spawned_names:
            try:
                self.spawned_names.remove(name)
            except ValueError:
                pass
        return bool(future.done() and future.exception() is None)

    def delete_all(self, timeout: float = 3.0):
        """嘗試刪除 self.spawned_names 的所有 instance（best-effort）"""
        for n in list(self.spawned_names):
            try:
                self.delete_object(n, timeout=timeout)
            except Exception:
                pass
        self.spawned_names = []

    # ---------- 隨機生成相關 ----------
    def get_safe_random_position(self, forbidden_positions: Optional[List[np.ndarray]] = None,
                                 min_dist: float = 0.08, max_attempts: int = 100) -> np.ndarray:
        """回傳 X/Y 在桌面內、且不落在手臂 XY 盒內的隨機 [x,y,z]，並盡量和 forbidden_positions 保持距離。"""
        arm = self.arm_bounds
        x0, x1 = X_RANGE
        y0, y1 = Y_RANGE
        forbidden_positions = forbidden_positions or []

        for _ in range(max_attempts):
            x = random.uniform(x0, x1)
            y = random.uniform(y0, y1)
            if arm is not None and (arm['x'][0] <= x <= arm['x'][1] and arm['y'][0] <= y <= arm['y'][1]):
                continue
            candidate = np.array([x, y, Z_HEIGHT], dtype=np.float32)
            # 距離檢查
            ok = True
            for p in forbidden_positions:
                if np.linalg.norm(candidate[:2] - np.asarray(p)[:2]) < min_dist:
                    ok = False
                    break
            if ok:
                return candidate
        # fallback：左下角
        return np.array([x0, y0, Z_HEIGHT], dtype=np.float32)

    def spawn_random_objects(self,
                             count: int,
                             *,
                             candidates: Optional[List[str]] = None,
                             avoid_overlap_dist: float = 0.08,
                             max_attempts_total: int = 500,
                             strict_unique: bool = True,
                             name_prefix: Optional[str] = None,
                             timeout: float = 10.0) -> List[Tuple[str, np.ndarray]]:
        """隨機 spawn 多個物件，無需手動指定位置。

        參數：
            count: 需要生成的物件數量。
            candidates: 可被隨機挑選的 base model 名稱清單（例如 ['mug','banana','coke_can']）。
                        若為 None 則使用建構子傳入的 self.object_names。
            avoid_overlap_dist: 盡量避免彼此距離小於此值（非嚴格）。
            max_attempts_total: 為每個 base 嘗試的總上限（包含重試）。
            strict_unique: 若 True，保證同一個 base model 只會被選一次（若 count 超過可用種類，會自動截斷到最大可用數量並給出警告）。
            name_prefix: 生成的 instance name 前綴；未填則使用 base 名稱 + uuid 的形式。
            timeout: 等待服務回應的秒數。

        回傳：[(instance_name, position), ...]（只包含 spawn 成功者）
        """
        if count <= 0:
            return []

        base_pool = list(candidates) if candidates is not None else list(self.object_names)
        if not base_pool:
            raise ValueError("No object names available. Provide candidates or set object_names in constructor.")

        # 準備 base 名單
        if strict_unique:
            if count > len(base_pool):
                self.get_logger().warn(
                    f"count({count}) > unique candidates({len(base_pool)}); trimming to {len(base_pool)} to keep unique.")
                count = len(base_pool)
            bases = random.sample(base_pool, k=count)
        else:
            bases = [random.choice(base_pool) for _ in range(count)]

        spawned: List[Tuple[str, np.ndarray]] = []
        forbidden_positions: List[np.ndarray] = []
        attempts = 0

        for base in bases:
            if attempts >= max_attempts_total:
                break
            attempts += 1
            pos = self.get_safe_random_position(forbidden_positions=forbidden_positions, min_dist=avoid_overlap_dist)
            inst_name = self._compose_instance_name(base, prefix=name_prefix)
            ok = self._spawn_via_service(base, pos, inst_name=inst_name, timeout=timeout)
            if ok:
                spawned.append((inst_name, pos))
                forbidden_positions.append(pos)
            # 若失敗，允許幾次重試（換位置與名字）
            retry = 0
            while (not ok) and retry < 3 and attempts < max_attempts_total:
                attempts += 1; retry += 1
                pos = self.get_safe_random_position(forbidden_positions=forbidden_positions, min_dist=avoid_overlap_dist)
                inst_name = self._compose_instance_name(base, prefix=name_prefix)
                ok = self._spawn_via_service(base, pos, inst_name=inst_name, timeout=timeout)
                if ok:
                    spawned.append((inst_name, pos))
                    forbidden_positions.append(pos)

        return spawned

    # ---------- 工具 ----------
    def list_spawned(self) -> List[str]:
        return list(self.spawned_names)

    # ---------- 私有：真正呼叫 SpawnEntity 服務 ----------
    def _compose_instance_name(self, base_name: str, prefix: Optional[str] = None) -> str:
        short = uuid.uuid4().hex[:8]
        if prefix:
            return f"{prefix}_{base_name}_{short}"
        return f"{base_name}_{short}"

    def _spawn_via_service(self, base_name: str, position: np.ndarray, *, inst_name: Optional[str] = None,
                           timeout: float = 10.0) -> bool:
        pos = np.asarray(position).reshape(3,)
        inst = inst_name or self._compose_instance_name(base_name)

        req = SpawnEntity.Request()
        req.name = inst
        req.xml = f"""
        <sdf version='1.6'>
          <model name='{inst}'>
            <include>
              <uri>model://{base_name}</uri>
            </include>
          </model>
        </sdf>
        """
        req.robot_namespace = inst
        req.initial_pose.position.x = float(pos[0])
        req.initial_pose.position.y = float(pos[1])
        req.initial_pose.position.z = float(pos[2])
        req.initial_pose.orientation.w = 1.0

        future = self.spawn_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        ok = future.done() and future.exception() is None and getattr(future.result(), 'success', False)

        if not ok:
            # 若失敗：嘗試刪除已存在的同名（安全退路）再重試一次
            try:
                self.delete_object(inst)
            except Exception:
                pass
            future = self.spawn_cli.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
            ok = future.done() and future.exception() is None and getattr(future.result(), 'success', False)

        if ok:
            self.spawned_names.append(inst)
        return bool(ok)
