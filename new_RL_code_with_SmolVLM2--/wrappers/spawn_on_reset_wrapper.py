from __future__ import annotations
try:
    import gymnasium as gym
except Exception:
    import gym
import os

class SpawnOnResetWrapper(gym.Wrapper):
    """在 reset 時，強制以固定數量重生桌面物件；episode 進行中不再動它。
    - 物件數量由環境變數 N_OBJECTS 控制（預設 3）。
    - 如需在 episode 途中補齊（不建議），可設 TOP_UP_OBJECTS=1。
    - 需要 env.spawner 物件，且 spawner 具有 delete_by_prefix / spawn_random_objects 之類 API。
    """
    def __init__(self, env: gym.Env, prefix: str = "obj", n_objects: int | None = None, top_up: bool = False):
        super().__init__(env)
        self.prefix = prefix
        self.n_objects = int(os.environ.get("N_OBJECTS", str(n_objects if n_objects is not None else 3)))
        # 預設不補齊，避免你看到數量在中途忽多忽少
        env_topup = os.environ.get("TOP_UP_OBJECTS", None)
        self.top_up = (env_topup is not None and env_topup not in ("0", "false", "False")) or bool(top_up)
        self._spawned_names: list[str] = []

    # ---------- internal helpers ----------
    def _delete_prefix(self):
        sp = getattr(self.env, "spawner", None)
        if sp and hasattr(sp, "delete_by_prefix"):
            try:
                sp.delete_by_prefix(self.prefix)
            except Exception:
                pass
        elif sp and hasattr(sp, "delete"):
            for name in list(self._spawned_names):
                try:
                    sp.delete(name)
                except Exception:
                    pass
        self._spawned_names.clear()

    def _count_existing(self) -> int:
        objs = getattr(self.env, "objects", None)
        if isinstance(objs, (list, tuple)):
            try:
                return sum(1 for o in objs if hasattr(o, "name") and str(o.name).startswith(self.prefix))
            except Exception:
                return len(self._spawned_names)
        return len(self._spawned_names)

    def _spawn_exact(self, n: int):
        if n <= 0:
            return
        sp = getattr(self.env, "spawner", None)
        if sp is None:
            return
        names: list[str] = []
        # 優先走你現有的 API
        if hasattr(sp, "spawn_random_objects"):
            try:
                spawned = sp.spawn_random_objects(num_objects=n, prefix=self.prefix, ensure_in_table=True, avoid_arm=True)
                for it in spawned:
                    name = it[0] if isinstance(it, (tuple, list)) and len(it) > 0 else str(it)
                    names.append(name)
            except Exception:
                pass
        elif hasattr(sp, "spawn_exact"):
            try:
                names = list(sp.spawn_exact(n, prefix=self.prefix))
            except Exception:
                pass
        self._spawned_names = names
        # 嘗試同步 env.objects（如果存在）
        objs = getattr(self.env, "objects", None)
        if isinstance(objs, list):
            try:
                from objects.sim_object import SimObject
                objs[:] = [SimObject(name=nm, position=None) for nm in names]
            except Exception:
                pass

    # ---------- gym API ----------
    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        # reset 後統一刪除舊前綴物體，再精準重生固定數量
        self._delete_prefix()
        self._spawn_exact(self.n_objects)
        return out

    def step(self, action):
        out = self.env.step(action)
        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
        else:
            obs, reward, done, info = out
            terminated, truncated = bool(done), False
        # 預設不補齊，避免中途數量變動；需要時才打開 TOP_UP_OBJECTS=1
        if self.top_up and not (terminated or truncated):
            try:
                cnt = self._count_existing()
                if cnt < self.n_objects:
                    self._spawn_exact(self.n_objects - cnt)
            except Exception:
                pass
        return out