"""Local person tracking: per-frame PersonDetections -> local track observations.

Boundary: `PersonTracker.update(detections, frame_bgr=None) -> list[TrackedDetection]`,
called once per processed frame, in frame order. `frame_bgr` is optional so motion-only
trackers can ignore it; appearance-gated trackers use the detector crop from that frame.

`local_track_id` is local to one tracker run over one video. It says nothing about who
the person is: ids restart from "1" in a new tracker (or after `reset()`), so the same id
in two videos is unrelated, and one person who leaves and returns may get a new id.

Each TrackedDetection carries the detector's own box and score for that frame, never the
tracker's predicted or smoothed box. A track that is only predicted in a frame (no
detection matched to it) produces nothing for that frame.

`ByteTrackPersonTracker` uses supervision's ByteTrack. supervision marks that class as
deprecated (removal planned for supervision 0.31); the version is pinned in requirements.

`RejectAndSplitPersonTracker` wraps ByteTrack with a cheap HSV-histogram continuity gate:
when ByteTrack would extend a track but the new crop disagrees with the track's signature,
a new local id is emitted instead (reject-and-split). It does not merge fragments and is
not ReID / P-codes.

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

import cv2
import numpy as np

from perception_minimal.detector import BBox, PersonDetection

# ByteTrack settings, equal to supervision 0.30.9's defaults (not tuned for any site).
BYTETRACK_SETTINGS = {
    "track_activation_threshold": 0.25,
    "lost_track_buffer": 30,
    "minimum_matching_threshold": 0.8,
    "minimum_consecutive_frames": 1,
}

# Local continuity gate (HSV histogram). Not a global identity embedding.
REJECT_AND_SPLIT_SETTINGS = {
    "hist_h_bins": 16,
    "hist_s_bins": 16,
    "hist_v_bins": 8,
    "min_correlation": 0.45,
    "ema_alpha": 0.35,
    "min_crop_side": 4,
    "min_chroma_pixels": 32,
    "min_saturation": 25,
    "min_value": 25,
}

_UNLINKED = 1e6  # assignment cost for a track/detection pair that cannot be the matched pair


@dataclass(frozen=True)
class TrackedDetection:
    local_track_id: str  # local to this tracker run / video; not a person identity
    bbox: BBox  # the detector's box, unchanged: x, y, width, height
    detection_score: float  # the detector's score, unchanged


class PersonTracker(Protocol):
    """Called once per processed frame, in order, with that frame's person detections.

    `frame_bgr` is the source BGR frame for this update when available (used by appearance
    gates). Motion-only implementations may ignore it.
    """

    def update(
        self,
        detections: Sequence[PersonDetection],
        frame_bgr: np.ndarray | None = None,
    ) -> list[TrackedDetection]: ...


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

    def update(
        self,
        detections: Sequence[PersonDetection],
        frame_bgr: np.ndarray | None = None,
    ) -> list[TrackedDetection]:
        del frame_bgr  # motion-only; appearance gates wrap this class
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


def _clip_bbox_to_frame(bbox: BBox, frame_bgr: np.ndarray) -> BBox | None:
    """Return the bbox clipped to the frame, or None if nothing remains."""
    h_img, w_img = frame_bgr.shape[:2]
    x, y, w, h = bbox
    x1 = max(0, int(x))
    y1 = max(0, int(y))
    x2 = min(w_img, int(x) + int(w))
    y2 = min(h_img, int(y) + int(h))
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None
    return (x1, y1, x2 - x1, y2 - y1)


def hsv_histogram(
    frame_bgr: np.ndarray,
    bbox: BBox,
    *,
    h_bins: int = REJECT_AND_SPLIT_SETTINGS["hist_h_bins"],
    s_bins: int = REJECT_AND_SPLIT_SETTINGS["hist_s_bins"],
    v_bins: int = REJECT_AND_SPLIT_SETTINGS["hist_v_bins"],
    min_crop_side: int = REJECT_AND_SPLIT_SETTINGS["min_crop_side"],
    min_chroma_pixels: int = REJECT_AND_SPLIT_SETTINGS["min_chroma_pixels"],
    min_saturation: int = REJECT_AND_SPLIT_SETTINGS["min_saturation"],
    min_value: int = REJECT_AND_SPLIT_SETTINGS["min_value"],
) -> np.ndarray | None:
    """L1-normalised HSV histogram of chromatic pixels in the detector crop.

    Returns None when the crop is too small or mostly grey/black/white. Callers treat
    None as \"no appearance evidence\" and do not split on that observation.
    """
    clipped = _clip_bbox_to_frame(bbox, frame_bgr)
    if clipped is None:
        return None
    x, y, w, h = clipped
    if w < min_crop_side or h < min_crop_side:
        return None
    crop = frame_bgr[y : y + h, x : x + w]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    chroma = (hsv[:, :, 1] >= min_saturation) & (hsv[:, :, 2] >= min_value)
    if int(chroma.sum()) < min_chroma_pixels:
        return None
    mask = chroma.astype(np.uint8) * 255
    hist = cv2.calcHist([hsv], [0, 1, 2], mask, [h_bins, s_bins, v_bins], [0, 180, 0, 256, 0, 256])
    hist = hist.astype(np.float32).ravel()
    total = float(hist.sum())
    if total <= 0:
        return None
    hist /= total
    return hist


def histogram_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """cv2 HISTCMP_CORREL on two flat L1-normalised histograms."""
    return float(cv2.compareHist(a.astype(np.float32), b.astype(np.float32), cv2.HISTCMP_CORREL))


class RejectAndSplitPersonTracker:
    """ByteTrack associator plus an HSV reject-and-split continuity gate.

    ByteTrack still proposes the association. This wrapper keeps a local HSV signature
    per emitted local track id. If a proposed continuation disagrees with that signature,
    a **new** local id is emitted and the ByteTrack id is remapped to it. Fragments are
    never merged back. Without `frame_bgr`, behaviour matches the inner tracker (ids pass
    through unchanged).
    """

    def __init__(
        self,
        frame_rate: float,
        *,
        inner: PersonTracker | None = None,
        min_correlation: float = REJECT_AND_SPLIT_SETTINGS["min_correlation"],
        ema_alpha: float = REJECT_AND_SPLIT_SETTINGS["ema_alpha"],
    ):
        if not math.isfinite(min_correlation) or not (-1.0 <= min_correlation <= 1.0):
            raise ValueError(f"min_correlation must be in [-1, 1], got {min_correlation!r}")
        if not math.isfinite(ema_alpha) or not (0.0 < ema_alpha <= 1.0):
            raise ValueError(f"ema_alpha must be in (0, 1], got {ema_alpha!r}")
        self.frame_rate = check_frame_rate(frame_rate)
        self._inner = inner if inner is not None else ByteTrackPersonTracker(self.frame_rate)
        self.min_correlation = float(min_correlation)
        self.ema_alpha = float(ema_alpha)
        self._signatures: dict[str, np.ndarray] = {}
        self._bt_to_local: dict[str, str] = {}
        self._next_local = 1
        inner_config = getattr(self._inner, "config", {"class": type(self._inner).__name__})
        self.config = {
            "tracker": "RejectAndSplitPersonTracker",
            "frame_rate": self.frame_rate,
            "gate": "reject_and_split_hsv",
            **REJECT_AND_SPLIT_SETTINGS,
            "min_correlation": self.min_correlation,
            "ema_alpha": self.ema_alpha,
            "inner": inner_config,
        }

    def reset(self) -> None:
        """Forget all tracks and signatures; ids restart from \"1\"."""
        reset = getattr(self._inner, "reset", None)
        if callable(reset):
            reset()
        self._signatures.clear()
        self._bt_to_local.clear()
        self._next_local = 1

    def _alloc_local(self) -> str:
        local = str(self._next_local)
        self._next_local += 1
        return local

    def update(
        self,
        detections: Sequence[PersonDetection],
        frame_bgr: np.ndarray | None = None,
    ) -> list[TrackedDetection]:
        proposed = self._inner.update(detections, frame_bgr)
        if frame_bgr is None:
            return proposed
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError(f"frame_bgr must be HxWx3, got shape {getattr(frame_bgr, 'shape', None)}")

        out: list[TrackedDetection] = []
        for t in proposed:
            hist = hsv_histogram(frame_bgr, t.bbox)
            bt_id = t.local_track_id
            local_id = self._bt_to_local.get(bt_id)

            if local_id is None:
                local_id = self._alloc_local()
                self._bt_to_local[bt_id] = local_id
                if hist is not None:
                    self._signatures[local_id] = hist
            else:
                prev = self._signatures.get(local_id)
                if hist is not None and prev is not None:
                    if histogram_correlation(prev, hist) < self.min_correlation:
                        local_id = self._alloc_local()
                        self._bt_to_local[bt_id] = local_id
                        self._signatures[local_id] = hist
                    else:
                        blended = (1.0 - self.ema_alpha) * prev + self.ema_alpha * hist
                        total = float(blended.sum())
                        if total > 0:
                            self._signatures[local_id] = blended / total
                elif hist is not None and prev is None:
                    self._signatures[local_id] = hist

            out.append(TrackedDetection(local_id, t.bbox, t.detection_score))

        out.sort(key=lambda t: int(t.local_track_id))
        if len({t.local_track_id for t in out}) != len(out):
            raise RuntimeError("RejectAndSplit emitted the same local track id twice in one frame")
        return out
