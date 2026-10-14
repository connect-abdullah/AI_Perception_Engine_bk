"""Person detection on single BGR frames.

Boundary: `PersonDetector.detect(frame_bgr) -> list[PersonDetection]`. Every detection
carries only a box and the detector's score; nothing about identity or tracking.

Box convention (the same as docs/reid_handoff_contract.md): `bbox = (x, y, width, height)`
in integer pixels of the original input frame, (x, y) the top-left corner. Boxes are
clipped to the frame, so 0 <= x, x + width <= frame width, 0 <= y, y + height <= frame
height, and width, height >= 1.

`TorchvisionPersonDetector` uses torchvision's Faster R-CNN ResNet-50 FPN v2 with its
pretrained COCO weights. These weights are local development / proof-of-concept
infrastructure only; they are not cleared for commercial or production use.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Protocol

import numpy as np

BBox = tuple[int, int, int, int]  # x, y, width, height


@dataclass(frozen=True)
class PersonDetection:
    bbox: BBox  # x, y, width, height; integer pixels of the original frame, clipped to it
    score: float  # the detector's raw score for "this box contains a person"; not identity evidence


class PersonDetector(Protocol):
    """Anything that finds people in one BGR uint8 frame of shape (H, W, 3)."""

    def detect(self, frame_bgr: np.ndarray) -> list[PersonDetection]: ...


def check_score_threshold(value) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"score_threshold must be a finite number in [0, 1], got {value!r}")
    return float(value)


def check_frame(frame_bgr) -> None:
    if not isinstance(frame_bgr, np.ndarray):
        raise ValueError(f"Expected a numpy BGR image, got {type(frame_bgr).__name__}")
    if frame_bgr.dtype != np.uint8 or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError(f"Expected a uint8 BGR image of shape (H, W, 3), got {frame_bgr.dtype} {frame_bgr.shape}")
    if frame_bgr.shape[0] == 0 or frame_bgr.shape[1] == 0:
        raise ValueError(f"Image is empty: shape {frame_bgr.shape}")


def xyxy_to_clipped_xywh(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> BBox | None:
    """Clip a float corner box to a `width` x `height` frame; return the smallest integer
    (x, y, width, height) box containing it, or None if nothing of it lies inside the frame."""
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
        return None
    x1, x2 = min(max(x1, 0.0), width), min(max(x2, 0.0), width)
    y1, y2 = min(max(y1, 0.0), height), min(max(y2, 0.0), height)
    if x2 <= x1 or y2 <= y1:
        return None
    left, top = math.floor(x1), math.floor(y1)
    right, bottom = math.ceil(x2), math.ceil(y2)
    return (left, top, right - left, bottom - top)


def select_person_detections(boxes_xyxy, labels, scores, person_label: int, score_threshold: float,
                             frame_width: int, frame_height: int) -> list[PersonDetection]:
    """Keep detections with label == person_label and score >= score_threshold, convert their
    boxes to clipped integer xywh, and sort by score (descending), then by box."""
    boxes_xyxy = np.asarray(boxes_xyxy, dtype=np.float64).reshape(-1, 4)
    labels = np.asarray(labels).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if not len(boxes_xyxy) == len(labels) == len(scores):
        raise ValueError(f"Mismatched detector output: {len(boxes_xyxy)} boxes, {len(labels)} labels, {len(scores)} scores")

    detections = []
    for box, label, score in zip(boxes_xyxy, labels, scores):
        if int(label) != person_label or not score >= score_threshold:
            continue
        bbox = xyxy_to_clipped_xywh(*box.tolist(), frame_width, frame_height)
        if bbox is not None:
            detections.append(PersonDetection(bbox, float(score)))
    detections.sort(key=lambda d: (-d.score, d.bbox))
    return detections


class TorchvisionPersonDetector:
    """Faster R-CNN ResNet-50 FPN v2 (torchvision, COCO weights), person class only.

    `score_threshold` is required: detections scoring below it are dropped. It is passed
    to the model as well, replacing torchvision's built-in 0.05 cut-off. Other model
    settings (NMS IoU, at most `box_detections_per_img` boxes per image across all
    classes, internal resizing) stay at torchvision's defaults and are listed in
    `provenance`. The model resizes internally but returns boxes in the coordinates of
    the frame passed in.

    Weights are fetched through the torchvision weights API into the torch hub cache
    (downloaded on first use).
    """

    def __init__(self, score_threshold: float, *, device: str = "cpu"):
        import torch
        import torchvision
        from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2

        self.score_threshold = check_score_threshold(score_threshold)
        self._torch = torch
        self.weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
        self.person_label = self.weights.meta["categories"].index("person")
        self.device = torch.device(device)
        self._to_input = self.weights.transforms()
        self._model = fasterrcnn_resnet50_fpn_v2(weights=self.weights, box_score_thresh=self.score_threshold)
        self._model.eval().to(self.device)

        roi, transform = self._model.roi_heads, self._model.transform
        self.provenance = {
            "detector": "torchvision.models.detection.fasterrcnn_resnet50_fpn_v2",
            "weights": f"{type(self.weights).__name__}.{self.weights.name}",
            "weights_url": self.weights.url,
            "weights_cache_path": str(Path(torch.hub.get_dir()) / "checkpoints" / Path(self.weights.url).name),
            "training_data": "COCO (train2017); development/PoC use only, not commercially cleared",
            "torch_version": torch.__version__,
            "torchvision_version": torchvision.__version__,
            "person_label": self.person_label,
            "score_threshold": self.score_threshold,
            "nms_iou": roi.nms_thresh,
            "box_detections_per_img": roi.detections_per_img,
            "internal_resize_min_size": transform.min_size,
            "internal_resize_max_size": transform.max_size,
            "device": str(self.device),
        }

    def detect(self, frame_bgr: np.ndarray) -> list[PersonDetection]:
        check_frame(frame_bgr)
        height, width = frame_bgr.shape[:2]
        rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])  # a copy; the caller's frame is untouched
        image = self._to_input(self._torch.from_numpy(rgb).permute(2, 0, 1)).to(self.device)
        with self._torch.inference_mode():
            output = self._model([image])[0]
        return select_person_detections(
            output["boxes"].cpu().numpy(),
            output["labels"].cpu().numpy(),
            output["scores"].cpu().numpy(),
            self.person_label,
            self.score_threshold,
            width,
            height,
        )
