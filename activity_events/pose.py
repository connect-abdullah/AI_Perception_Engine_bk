"""Pose on person crops and a simple posture rule (sitting / standing / unknown).

The pose model is behind a small seam (`PoseEstimator`) so it can be swapped later
(another model, or a pose layer from the 3D work) without touching the activity logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

# COCO-17 keypoint indices
L_SHO, R_SHO, L_HIP, R_HIP, L_KNEE, R_KNEE = 5, 6, 11, 12, 13, 14


@dataclass(frozen=True)
class Pose:
    keypoints: np.ndarray  # (17, 2) pixel coordinates in the crop
    confidence: np.ndarray  # (17,) per-keypoint confidence in [0, 1]


class PoseEstimator(Protocol):
    name: str

    def estimate(self, crop_path: Path) -> Pose | None:
        """Pose of the main person in the crop, or None if no person/pose is found."""


@dataclass(frozen=True)
class PostureSettings:
    min_keypoint_confidence: float = 0.5
    # knee drop below the hip, divided by torso length. Standing ~0.8-1.1, sitting
    # (thighs roughly horizontal) well below. Hand-tuned on webcam footage, not calibrated.
    sit_ratio: float = 0.55


def posture_from_pose(pose: Pose | None, settings: PostureSettings = PostureSettings()) -> str | None:
    """'sitting', 'standing', or None when shoulders/hips/knees are not visible enough."""
    if pose is None:
        return None
    kps, conf = pose.keypoints, pose.confidence
    ratios = []
    for sho, hip, knee in ((L_SHO, L_HIP, L_KNEE), (R_SHO, R_HIP, R_KNEE)):
        if min(conf[sho], conf[hip], conf[knee]) < settings.min_keypoint_confidence:
            continue
        torso = float(np.linalg.norm(kps[hip] - kps[sho]))
        if torso < 1.0:
            continue
        ratios.append((kps[knee][1] - kps[hip][1]) / torso)  # image y grows downwards
    if not ratios:
        return None
    return "sitting" if float(np.mean(ratios)) < settings.sit_ratio else "standing"


class YoloPoseEstimator:
    """Ultralytics YOLO-pose on a single crop; keeps the largest detected person.

    Crops are often small (far-away people), so they are upscaled before inference.
    """

    def __init__(self, model: str = "yolo11n-pose.pt", device: str | None = None, min_side: int = 256):
        from ultralytics import YOLO  # imported lazily: tests and --no-pose runs do not need it

        self.name = f"ultralytics:{model}"
        self._model = YOLO(model)
        self._device = device
        self._min_side = min_side

    def estimate(self, crop_path: Path) -> Pose | None:
        import cv2

        img = cv2.imread(str(crop_path))
        if img is None or img.size == 0:
            return None
        h, w = img.shape[:2]
        scale = max(1.0, self._min_side / max(1, min(h, w)))
        if scale > 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
        res = self._model.predict(img, verbose=False, device=self._device)[0]
        if res.keypoints is None or res.boxes is None or len(res.boxes) == 0:
            return None
        boxes = res.boxes.xyxy.cpu().numpy()
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        i = int(np.argmax(areas))
        kxy = res.keypoints.xy.cpu().numpy()[i] / scale
        kconf = (
            res.keypoints.conf.cpu().numpy()[i]
            if res.keypoints.conf is not None
            else np.ones(kxy.shape[0], dtype=np.float32)
        )
        return Pose(keypoints=kxy.astype(np.float32), confidence=kconf.astype(np.float32))
