"""
Gazebo Object Spawner (ROS2) — v2
---------------------------------
功能：
- 透過 Gazebo 的 `/spawn_entity`、`/delete_entity`、`/get_model_list` 服務，將物件 (URDF/SDF/XML字串/檔案) spawn 進世界
- 提供「不重疊」的桌面隨機擺放：在 `table_area` 內根據半徑 `radius` 迴避重疊與邊界
- 回傳 spawn 結果（名稱、姿態、半徑），方便在 `reset()` 時放進 `info`

設計重點：
- 不跟 wrapper 合併；單純專注在 spawn / delete / list 與簡單 layout 規劃
- 命名若重複，會自動加後綴 `_1`, `_2`, ...
- 支援從檔案讀取（.urdf / .sdf / .xml），或直接給 XML 字串

需求：
- 服務存在：/spawn_entity, /delete_entity, /get_model_list
  (你的環境清單已包含這三個服務)

用法範例：
```python
from objects.spawner import Spawner, SpawnerConfig, TableArea, ModelSpec

sp = Spawner(SpawnerConfig(table_area=TableArea(xmin=-0.3, xmax=0.3, ymin=0.2, ymax=0.8, z=0.76)))
sp.wait_until_ready()

specs = [
    ModelSpec(name="cube", file_path="/home/user/models/cube.urdf", fmt="urdf", radius=0.035),
    ModelSpec(name="mug",  file_path="/home/user/models/mug.sdf",  fmt="sdf",  radius=0.045),
]
spawned = sp.spawn_batch(specs, randomize=True, seed=42)
# spawned -> list of SpawnedModel(name, xyzrpy=(x,y,z,roll,pitch,yaw), radius)
```
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Iterable
import os
import time

import numpy as np

import rclpy
from rclpy.node import Node

# Gazebo services
from geometry_msgs.msg import Pose, Point, Quaternion
from gazebo_msgs.srv import SpawnEntity, DeleteEntity, GetModelList


# ----------------------------- Data -----------------------------

@dataclass
class TableArea:
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    z: float  # table top Z for placing objects (object base contact)

    def contains(self, x: float, y: float, margin: float = 0.0) -> bool:
        return (self.xmin + margin <= x <= self.xmax - margin) and (self.ymin + margin <= y <= self.ymax - margin)


@dataclass
class SpawnerConfig:
    table_area: TableArea
    name_prefix: str = "obj"
    max_spawn_tries: int = 200
    separation_margin: float = 0.01  # extra distance added on top of (r_i + r_j)
    default_roll_pitch: Tuple[float, float] = (0.0, 0.0)
    default_yaw_range: Tuple[float, float] = (-np.pi, np.pi)
    reference_frame: str = "world"
    wait_timeout: float = 3.0


@dataclass
class ModelSpec:
    name: str
    # 一下兩者擇一
    xml: Optional[str] = None      # 直接給 XML 字串（URDF/SDF）
    file_path: Optional[str] = None  # 從檔案讀（支援 .urdf / .sdf / .xml）
    fmt: str = "sdf"              # "urdf" 或 "sdf"，僅用於註記與除錯
    radius: float = 0.04           # 約略圓盤半徑（用於避免重疊）
    z_offset: float = 0.0          # 物件自身模型原點到桌面的高度偏移（需要時填）
    yaw_random: bool = True
    yaw_range: Optional[Tuple[float, float]] = None  # 若不給，採用 config.default_yaw_range


@dataclass
class SpawnedModel:
    name: str
    xyzrpy: Tuple[float, float, float, float, float, float]
    radius: float


# ----------------------------- Spawner -----------------------------

class Spawner(Node):
    def __init__(self, cfg: SpawnerConfig):
        super().__init__("tabletop_spawner")
        self.cfg = cfg
        self._cli_spawn = self.create_client(SpawnEntity, "/spawn_entity")
        self._cli_delete = self.create_client(DeleteEntity, "/delete_entity")
        self._cli_list = self.create_client(GetModelList, "/get_model_list")

    # -- lifecycle -----------------------------------------------------
    def wait_until_ready(self) -> bool:
        ok1 = self._cli_spawn.wait_for_service(timeout_sec=self.cfg.wait_timeout)
        ok2 = self._cli_delete.wait_for_service(timeout_sec=self.cfg.wait_timeout)
        ok3 = self._cli_list.wait_for_service(timeout_sec=self.cfg.wait_timeout)
        if not (ok1 and ok2 and ok3):
            self.get_logger().warn("Spawner services not all available.")
        return ok1 and ok2 and ok3

    # -- model list / delete ------------------------------------------
    def list_models(self) -> List[str]:
        fut = self._cli_list.call_async(GetModelList.Request())
        rclpy.spin_until_future_complete(self, fut, timeout_sec=self.cfg.wait_timeout)
        res = fut.result()
        return list(res.model_names) if res is not None else []

    def delete(self, name: str) -> bool:
        req = DeleteEntity.Request()
        req.name = name
        fut = self._cli_delete.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=self.cfg.wait_timeout)
        return fut.result() is not None

    def delete_many(self, names: Iterable[str]) -> int:
        cnt = 0
        for n in names:
            try:
                if self.delete(n):
                    cnt += 1
            except Exception:
                pass
        return cnt

    # -- spawn core ----------------------------------------------------
    def spawn(self, name: str, xml: str, xyzrpy: Tuple[float, float, float, float, float, float]) -> bool:
        x, y, z, r, p, yaw = xyzrpy
        pose = Pose()
        pose.position = Point(x=float(x), y=float(y), z=float(z))
        # convert rpy -> quaternion (ZYX yaw-pitch-roll)
        qx, qy, qz, qw = self._rpy_to_quat(r, p, yaw)
        pose.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)

        # resolve duplicate names
        name_final = self._resolve_unique_name(name)

        req = SpawnEntity.Request()
        req.name = name_final
        req.xml = xml
        req.robot_namespace = ""
        req.initial_pose = pose
        req.reference_frame = self.cfg.reference_frame

        fut = self._cli_spawn.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=self.cfg.wait_timeout)
        ok = fut.result() is not None
        if not ok:
            self.get_logger().warn(f"spawn failed: {name_final}")
        return ok

    # -- batch with layout --------------------------------------------
    def spawn_batch(self, specs: List[ModelSpec], randomize: bool = True, seed: Optional[int] = None) -> List[SpawnedModel]:
        rng = np.random.default_rng(seed if seed is not None else int(time.time() * 1000) & 0xFFFFFFFF)
        placed: List[Tuple[float, float, float]] = []  # (x, y, radius)
        out: List[SpawnedModel] = []

        for spec in specs:
            # resolve XML
            xml = self._resolve_xml(spec)
            if xml is None:
                self.get_logger().warn(f"skip spec without xml: {spec.name}")
                continue

            # sample layout
            if randomize:
                xy = self._sample_xy_nonoverlap(rng, spec.radius, placed)
            else:
                # deterministic center drop (best-effort)
                cx = 0.5 * (self.cfg.table_area.xmin + self.cfg.table_area.xmax)
                cy = 0.5 * (self.cfg.table_area.ymin + self.cfg.table_area.ymax)
                xy = np.array([cx, cy], dtype=np.float32)

            yaw = self._sample_yaw(rng, spec) if spec.yaw_random else 0.0
            roll, pitch = self.cfg.default_roll_pitch
            z = self.cfg.table_area.z + spec.z_offset
            xyzrpy = (float(xy[0]), float(xy[1]), float(z), float(roll), float(pitch), float(yaw))

            ok = self.spawn(spec.name, xml, xyzrpy)
            if ok:
                placed.append((float(xy[0]), float(xy[1]), float(spec.radius)))
                out.append(SpawnedModel(name=self._resolve_latest_name(spec.name), xyzrpy=xyzrpy, radius=float(spec.radius)))
        return out

    # ---------------------------- helpers ----------------------------
    def _resolve_xml(self, spec: ModelSpec) -> Optional[str]:
        if spec.xml is not None and spec.xml.strip():
            return spec.xml
        if spec.file_path is None:
            return None
        path = os.path.expanduser(spec.file_path)
        if not os.path.isfile(path):
            # 允許非本機路徑（如 package://），交給 Gazebo 處理；這裡直接當作 xml 字串載入不行，回傳 None
            # 若你有 xacro，請先外部轉換成 urdf 再給進來
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception:
                return None
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def _sample_yaw(self, rng: np.random.Generator, spec: ModelSpec) -> float:
        lo, hi = spec.yaw_range if spec.yaw_range is not None else self.cfg.default_yaw_range
        return float(rng.uniform(lo, hi))

    def _sample_xy_nonoverlap(self, rng: np.random.Generator, radius: float, placed: List[Tuple[float, float, float]]) -> np.ndarray:
        ar = self.cfg.table_area
        tries = 0
        margin = float(radius + self.cfg.separation_margin)
        while tries < self.cfg.max_spawn_tries:
            x = float(rng.uniform(ar.xmin + margin, ar.xmax - margin))
            y = float(rng.uniform(ar.ymin + margin, ar.ymax - margin))
            if not ar.contains(x, y, margin=margin):
                tries += 1
                continue
            ok = True
            for (px, py, pr) in placed:
                d = float(np.hypot(x - px, y - py))
                if d < (margin + pr):
                    ok = False
                    break
            if ok:
                return np.array([x, y], dtype=np.float32)
            tries += 1
        # fallback：放中間（可能重疊，之後靠 RL 自行解決）
        cx = 0.5 * (ar.xmin + ar.xmax)
        cy = 0.5 * (ar.ymin + ar.ymax)
        return np.array([cx, cy], dtype=np.float32)

    def _rpy_to_quat(self, roll: float, pitch: float, yaw: float) -> Tuple[float, float, float, float]:
        cr = np.cos(roll * 0.5); sr = np.sin(roll * 0.5)
        cp = np.cos(pitch * 0.5); sp = np.sin(pitch * 0.5)
        cy = np.cos(yaw * 0.5); sy = np.sin(yaw * 0.5)
        qw = cr*cp*cy + sr*sp*sy
        qx = sr*cp*cy - cr*sp*sy
        qy = cr*sp*cy + sr*cp*sy
        qz = cr*cp*sy - sr*sp*cy
        return float(qx), float(qy), float(qz), float(qw)

    def _resolve_unique_name(self, base: str) -> str:
        existing = set(self.list_models())
        if base not in existing:
            self._last_name = base
            return base
        i = 1
        while f"{base}_{i}" in existing:
            i += 1
        self._last_name = f"{base}_{i}"
        return self._last_name

    def _resolve_latest_name(self, base: str) -> str:
        return getattr(self, "_last_name", base)


__all__ = [
    "TableArea",
    "SpawnerConfig",
    "ModelSpec",
    "SpawnedModel",
    "Spawner",
]
