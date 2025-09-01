# wrappers/spawn_on_reset_wrapper.py
from __future__ import annotations
from typing import Iterable, Tuple

try:
    import gymnasium as gym
except Exception:
    import gym

import numpy as np

from objects.spawner import Spawner


class SpawnOnResetWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env,
                 object_names: Iterable[str] = ("beer", "mug", "coke_can"),
                 count_range: Tuple[int, int] = (2, 4),
                 avoid_overlap_dist: float = 0.10,
                 strict_unique: bool = False):
        super().__init__(env)
        self.object_names = list(object_names)
        self.count_range = tuple(count_range)
        self.avoid_overlap_dist = float(avoid_overlap_dist)
        self.strict_unique = bool(strict_unique)
        # 單獨的 Spawner 節點，避免動到原 env
        self._spawner = Spawner(object_names=self.object_names)

    def reset(self, **kwargs):
        try:
            self._spawner.delete_all()
        except Exception:
            pass
        try:
            n = int(np.random.randint(self.count_range[0], self.count_range[1] + 1))
            self._spawner.spawn_random_objects(
                count=n,
                candidates=self.object_names,
                avoid_overlap_dist=self.avoid_overlap_dist,
                strict_unique=self.strict_unique,
                name_prefix="ep",
            )
        except Exception:
            pass
        return self.env.reset(**kwargs)

    def close(self):
        try:
            self._spawner.delete_all()
            self._spawner.destroy_node()
        except Exception:
            pass
        super().close()