# reward_engine.py
import numpy as np
import cv2
from vlm import VLMScorer


class ProximityReward:
    def __init__(self, max_dist=0.3, min_dist=0.05):
        self.max_dist = max_dist
        self.min_dist = min_dist
        self.prev_d = float("inf")

    def reset(self):
        self.prev_d = float("inf")

    def compute(self, current_d: float) -> float:
        if current_d >= self.max_dist:
            return 0.0
        current_d = max(current_d, self.min_dist)
        delta = self.prev_d - current_d
        shaped = max(0.0, delta / (self.max_dist - self.min_dist))
        self.prev_d = current_d
        return shaped


class Zone:
    def __init__(self, center: tuple[float, float], size: tuple[float, float]):
        self.cx, self.cy = center
        self.w, self.h = size

    def inside(self, x, y, tol=0.01):
        return (self.cx - self.w / 2 - tol <= x <= self.cx + self.w / 2 + tol and
                self.cy - self.h / 2 - tol <= y <= self.cy + self.h / 2 + tol)


class ZonePlaceReward:
    def __init__(self, zone: Zone):
        self.zone = zone
        self.placed_objects = set()

    def reset(self):
        self.placed_objects.clear()

    def compute(self, obj_name: str, obj_pos: tuple) -> float:
        if obj_name in self.placed_objects:
            return 0.0
        x, y, z = obj_pos
        if self.zone.inside(x, y):
            self.placed_objects.add(obj_name)
            return 0.8
        return 0.0


class GraspReward:
    def __init__(self):
        self.prev_attached = False

    def reset(self):
        self.prev_attached = False

    def compute(self, current_attached: bool) -> float:
        reward = 0.0
        if not self.prev_attached and current_attached:
            reward = 0.5
        self.prev_attached = current_attached
        return reward


class AlignmentReward:
    def __init__(self, min_area: int = 300, y_var_norm: float = 10.0):
        self.min_area = min_area
        self.y_var_norm = y_var_norm

    def compute(self, image: np.ndarray) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        centers_y = [y + h * 0.5 for x, y, w, h in (cv2.boundingRect(c) for c in contours) if w * h >= self.min_area]

        if len(centers_y) < 2:
            return 0.0

        y_var = float(np.var(centers_y))
        return max(0.0, 1.0 - (y_var / (self.y_var_norm ** 2)))


class VLMReward:
    def __init__(self, instruction: str = "align objects in a row"):
        self.scorer = VLMScorer()
        self.instruction = instruction
        self.prev_image = None

    def update_prev(self, image: np.ndarray):
        self.prev_image = image

    def compare(self, image: np.ndarray) -> float:
        if self.prev_image is None:
            self.update_prev(image)
            return 0.0

        winner = self.scorer.compare_images(self.prev_image, image, self.instruction)
        score = 1.0 if winner == "image_2" else -1.0 if winner == "image_1" else 0.0
        self.update_prev(image)
        return score * 0.5


class CombinedReward:
    def __init__(self, zone_cfg):
        zone = Zone(center=tuple(zone_cfg['center']), size=tuple(zone_cfg['size']))
        self.proximity = ProximityReward()
        self.grasp = GraspReward()
        self.place = ZonePlaceReward(zone)
        self.align = AlignmentReward()
        self.vlm = VLMReward()

    def reset(self):
        self.proximity.reset()
        self.grasp.reset()
        self.place.reset()

    def compute_step(self, dist: float, attached: bool, obj_name: str, obj_pos: tuple) -> float:
        r1 = self.proximity.compute(dist)
        r2 = self.grasp.compute(attached)
        r3 = self.place.compute(obj_name, obj_pos)
        return r1 + r2 + r3

    def compute_terminal(self, image: np.ndarray, success: bool) -> float:
        if not success:
            return 0.0
        return self.align.compute(image) + self.vlm.compare(image)
