"""Local person tracking: per-frame PersonDetections -> local track observations.

Boundary: `PersonTracker.update(detections) -> list[TrackedDetection]`, called once per
processed frame, in frame order.

`local_track_id` is local to one tracker run over one video. It says nothing about who
the person is: ids restart from "1" in a new tracker (or after `reset()`), so the same id
in two videos is unrelated, and one person who leaves and returns may get a new id.

Each TrackedDetection carries the detector's own box and score for that frame, never the
tracker's predicted or smoothed box. A track that is only predicted in a frame (no
detection matched to it) produces nothing for that frame.

`ByteTrackPersonTracker` uses supervision's ByteTrack. supervision marks that class as
deprecated (removal planned for supervision 0.31); the version is pinned in requirements.
Callers depend only on `PersonTracker` and `TrackedDetection`, so another tracker
implementation can replace it without changing them.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Protocol

import numpy as np

from perception_minimal.detector import BBox, PersonDetection

# ByteTrack settings, equal to supervision 0.30.9's defaults (not tuned for any site).
BYTETRACK_SETTINGS = {
    "track_activation_threshold": 0.25,
    "lost_track_buffer": 30,
    "minimum_matching_threshold": 0.8,
    "minimum_consecutive_frames": 1,
}

_UNLINKED = 1e6  # assignment cost for a track/detection pair that cannot be the matched pair


@dataclass(frozen=True)
class TrackedDetection:
    local_track_id: str  # local to this tracker run / video; not a person identity
    bbox: BBox  # the detector's box, unchanged: x, y, width, height
    detection_score: float  # the detector's score, unchanged


class PersonTracker(Protocol):
    """Called once per processed frame, in order, with that frame's person detections."""

    def update(self, detections: Sequence[PersonDetection]) -> list[TrackedDetection]: ...


def check_frame_rate(value) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"frame_rate must be a finite number > 0, got {value!r}")
    return float(value)


def _check_detections(detections) -> list[PersonDetection]:
    if isinstance(detections, (str, bytes)) or not isinstance(detections, Sequence):
        raise TypeError(f"detections must be a sequence of PersonDetection, got {type(detections).__name__}")
    for d in detections:
        if not isinstance(d, PersonDetection):
            raise TypeError(f"Expected PersonDetection, got {type(d).__name__}")
        x, y, w, h = d.bbox
        if w < 1 or h < 1:
            raise ValueError(f"Detection box must have positive size, got {d.bbox}")
    return list(detections)


def _iou(a, b) -> float:
    """IoU of two (x1, y1, x2, y2) boxes."""
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


class ByteTrackPersonTracker:
    """ByteTrack (supervision 0.30.9) over PersonDetections, one instance per video.

    `frame_rate` is the rate at which `update` is called: video fps / stride. ByteTrack
    keeps a lost track for int(frame_rate / 30 * lost_track_buffer) updates (about one
    second with the default buffer).

    ByteTrack behaviour that filters detections (library defaults, recorded in `config`):
      * detections scoring <= 0.1 are ignored;
      * a new track needs a detection scoring >= track_activation_threshold + 0.1;
      * detections between 0.1 and track_activation_threshold can only extend existing tracks;
      * after the first frame, a new track is confirmed on its second matched frame, and only
        then emitted; its first detection is not emitted.

    Linking a track to its detection: supervision's own `update_with_detections` re-pairs
    detections with tracks by IoU against the tracker's smoothed boxes. Instead, each
    emitted track is paired with the detection it was actually updated with: ByteTrack
    copies that detection's score onto the track, so only detections with exactly that
    score are candidates, and IoU decides only between candidates with identical scores.
    """

    def __init__(self, frame_rate: float):
        from supervision.tracker.byte_tracker.core import ByteTrack  # direct import, not the deprecated alias

        self.frame_rate = check_frame_rate(frame_rate)
        with warnings.catch_warnings():
            # Only the known deprecation notice for this pinned version is silenced.
            warnings.filterwarnings("ignore", message=r"The `ByteTrack` was deprecated", category=FutureWarning)
            self._tracker = ByteTrack(frame_rate=self.frame_rate, **BYTETRACK_SETTINGS)
        self.config = {
            "tracker": "supervision.tracker.byte_tracker.core.ByteTrack",
            "frame_rate": self.frame_rate,
            **BYTETRACK_SETTINGS,
            "max_frames_lost": self._tracker.max_time_lost,
            "new_track_min_score": self._tracker.det_thresh,
            "ignored_at_or_below_score": 0.1,
        }

    def reset(self) -> None:
        """Forget all tracks; ids restart from "1". Use between videos."""
        self._tracker.reset()

    def update(self, detections: Sequence[PersonDetection]) -> list[TrackedDetection]:
        detections = _check_detections(detections)
        boxes = np.array([(x, y, x + w, y + h) for x, y, w, h in (d.bbox for d in detections)], dtype=np.float64)
        scores = np.array([d.score for d in detections], dtype=np.float64)
        tensors = np.hstack([boxes.reshape(-1, 4), scores.reshape(-1, 1)])

        tracks = self._tracker.update_with_tensors(tensors)
        if not tracks:
            return []
        if not detections:
            raise RuntimeError("ByteTrack reported matched tracks for a frame without detections")

        from scipy.optimize import linear_sum_assignment

        cost = np.full((len(tracks), len(detections)), _UNLINKED)
        for i, track in enumerate(tracks):
            track_box = track.tlbr
            for j in range(len(detections)):
                if track.score == scores[j]:
                    cost[i, j] = 1.0 - _iou(track_box, boxes[j])
        rows, cols = linear_sum_assignment(cost)
        if len(rows) != len(tracks) or (cost[rows, cols] >= _UNLINKED).any():
            raise RuntimeError("Could not link every ByteTrack track to the detection it was updated with")

        out = [
            TrackedDetection(str(int(tracks[i].external_track_id)), detections[j].bbox, detections[j].score)
            for i, j in zip(rows, cols)
        ]
        out.sort(key=lambda t: int(t.local_track_id))
        if len({t.local_track_id for t in out}) != len(out):
            raise RuntimeError("ByteTrack emitted the same track id twice in one frame")
        return out
