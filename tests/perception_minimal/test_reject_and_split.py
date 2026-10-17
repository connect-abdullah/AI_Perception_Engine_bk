"""Reject-and-split HSV gate tests. Synthetic frames only; no detector weights."""

from __future__ import annotations

import numpy as np
import pytest

from perception_minimal.detector import PersonDetection
from perception_minimal.tracker import (
    REJECT_AND_SPLIT_SETTINGS,
    ByteTrackPersonTracker,
    RejectAndSplitPersonTracker,
    TrackedDetection,
    histogram_correlation,
    hsv_histogram,
)

FPS = 10.0


class StickyInner:
    """Always emits ByteTrack-style id \"1\" for the first detection (forces a chain)."""

    def __init__(self, frame_rate: float = FPS):
        self.frame_rate = frame_rate
        self.config = {"tracker": "StickyInner", "frame_rate": frame_rate}

    def reset(self) -> None:
        pass

    def update(self, detections, frame_bgr=None):
        del frame_bgr
        if not detections:
            return []
        d = detections[0]
        return [TrackedDetection("1", d.bbox, d.score)]


def paint(w, h, boxes_bgr):
    """HxW BGR frame with filled rectangles: list of (bbox, bgr_color)."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    for (x, y, bw, bh), color in boxes_bgr:
        frame[y : y + bh, x : x + bw] = color
    return frame


def test_hsv_histogram_distinguishes_colors():
    red = paint(80, 80, [((10, 10, 40, 60), (0, 0, 255))])
    blue = paint(80, 80, [((10, 10, 40, 60), (255, 0, 0))])
    hr = hsv_histogram(red, (10, 10, 40, 60))
    hb = hsv_histogram(blue, (10, 10, 40, 60))
    assert hr is not None and hb is not None
    assert histogram_correlation(hr, hr) > 0.99
    assert histogram_correlation(hr, hb) < REJECT_AND_SPLIT_SETTINGS["min_correlation"]


def test_same_appearance_keeps_one_local_id():
    tracker = RejectAndSplitPersonTracker(FPS, inner=StickyInner())
    ids = []
    for t in range(12):
        bbox = (20 + 2 * t, 30, 40, 60)
        frame = paint(200, 120, [(bbox, (40, 180, 40))])
        out = tracker.update([PersonDetection(bbox, 0.9)], frame)
        ids.append([x.local_track_id for x in out])
    assert all(i == ["1"] for i in ids)


def test_appearance_change_under_same_inner_id_splits():
    tracker = RejectAndSplitPersonTracker(FPS, inner=StickyInner(), min_correlation=0.45)
    green_bbox = (20, 30, 40, 60)
    blue_bbox = (24, 30, 40, 60)

    for _ in range(5):
        frame = paint(200, 120, [(green_bbox, (40, 180, 40))])
        out = tracker.update([PersonDetection(green_bbox, 0.9)], frame)
        assert [t.local_track_id for t in out] == ["1"]

    frame = paint(200, 120, [(blue_bbox, (255, 40, 40))])
    out = tracker.update([PersonDetection(blue_bbox, 0.9)], frame)
    assert [t.local_track_id for t in out] == ["2"]

    # Further frames of the blue person keep the new id (mapped from sticky "1").
    for _ in range(4):
        frame = paint(200, 120, [(blue_bbox, (255, 40, 40))])
        out = tracker.update([PersonDetection(blue_bbox, 0.9)], frame)
        assert [t.local_track_id for t in out] == ["2"]


def test_without_frame_passes_through_inner_ids():
    tracker = RejectAndSplitPersonTracker(FPS, inner=StickyInner())
    out = tracker.update([PersonDetection((10, 10, 20, 40), 0.9)])
    assert [t.local_track_id for t in out] == ["1"]


def test_does_not_merge_after_split():
    """After a split, the abandoned signature is not reused for a later similar crop."""
    tracker = RejectAndSplitPersonTracker(FPS, inner=StickyInner(), min_correlation=0.45)
    green = (20, 30, 40, 60)
    blue = (24, 30, 40, 60)

    for _ in range(3):
        tracker.update([PersonDetection(green, 0.9)], paint(200, 120, [(green, (40, 180, 40))]))
    tracker.update([PersonDetection(blue, 0.9)], paint(200, 120, [(blue, (255, 40, 40))]))  # -> "2"
    # Green returns under the same sticky inner id; must become a new local id, not "1" again.
    out = tracker.update([PersonDetection(green, 0.9)], paint(200, 120, [(green, (40, 180, 40))]))
    assert [t.local_track_id for t in out] == ["3"]


def test_wraps_real_bytetrack_stable_colored_walker():
    tracker = RejectAndSplitPersonTracker(FPS)
    seen = []
    for t in range(20):
        bbox = (10 + 3 * t, 30, 30, 50)
        frame = paint(200, 120, [(bbox, (200, 80, 80))])
        out = tracker.update([PersonDetection(bbox, 0.9)], frame)
        seen.extend(x.local_track_id for x in out)
    assert set(seen) == {"1"}


def test_reset_restarts_local_ids_and_signatures():
    tracker = RejectAndSplitPersonTracker(FPS, inner=StickyInner())
    bbox = (20, 30, 40, 60)
    frame = paint(200, 120, [(bbox, (40, 180, 40))])
    assert [t.local_track_id for t in tracker.update([PersonDetection(bbox, 0.9)], frame)] == ["1"]
    tracker.reset()
    assert [t.local_track_id for t in tracker.update([PersonDetection(bbox, 0.9)], frame)] == ["1"]


def test_config_records_gate_and_inner():
    tracker = RejectAndSplitPersonTracker(FPS)
    assert tracker.config["gate"] == "reject_and_split_hsv"
    assert tracker.config["tracker"] == "RejectAndSplitPersonTracker"
    assert "inner" in tracker.config
    assert tracker.config["min_correlation"] == REJECT_AND_SPLIT_SETTINGS["min_correlation"]


def test_bad_gate_params_rejected():
    with pytest.raises(ValueError, match="min_correlation"):
        RejectAndSplitPersonTracker(FPS, min_correlation=2.0)
    with pytest.raises(ValueError, match="ema_alpha"):
        RejectAndSplitPersonTracker(FPS, ema_alpha=0.0)


def test_default_pipeline_factory_is_reject_and_split():
    from perception_minimal.run import run_pipeline
    import inspect

    default = inspect.signature(run_pipeline).parameters["tracker_factory"].default
    assert default is RejectAndSplitPersonTracker


def test_settings_expose_chroma_defaults():
    """HSV gate chroma knobs are in REJECT_AND_SPLIT_SETTINGS (not hidden function defaults)."""
    for key in ("min_chroma_pixels", "min_saturation", "min_value"):
        assert key in REJECT_AND_SPLIT_SETTINGS
    tracker = RejectAndSplitPersonTracker(FPS)
    for key in ("min_chroma_pixels", "min_saturation", "min_value"):
        assert tracker.config[key] == REJECT_AND_SPLIT_SETTINGS[key]


class CrossingStickyInner:
    """Emits sticky ids \"1\" and \"2\" for up to two detections (simulates crossed associations)."""

    def __init__(self, frame_rate: float = FPS):
        self.frame_rate = frame_rate
        self.config = {"tracker": "CrossingStickyInner", "frame_rate": frame_rate}

    def reset(self) -> None:
        pass

    def update(self, detections, frame_bgr=None):
        del frame_bgr
        out = []
        for i, d in enumerate(detections[:2]):
            out.append(TrackedDetection(str(i + 1), d.bbox, d.score))
        return out


def test_crossing_different_colors_splits_swapped_association():
    """Two people cross: inner keeps id \"1\" on the wrong body (color change) → local split."""
    tracker = RejectAndSplitPersonTracker(FPS, inner=CrossingStickyInner(), min_correlation=0.45)
    # Person A (green) at left, person B (blue) at right — establish signatures on ids 1 and 2.
    left, right = (20, 30, 40, 60), (100, 30, 40, 60)
    for _ in range(4):
        frame = paint(200, 120, [(left, (40, 180, 40)), (right, (255, 40, 40))])
        out = tracker.update(
            [PersonDetection(left, 0.9), PersonDetection(right, 0.9)],
            frame,
        )
        assert {t.local_track_id for t in out} == {"1", "2"}

    # CrossingStickyInner maps detections[0]→\"1\", detections[1]→\"2\".
    # Feed blue first so sticky \"1\" (established on green) gets the wrong body → split.
    blue_box, green_box = (100, 30, 40, 60), (20, 30, 40, 60)
    frame = paint(200, 120, [(blue_box, (255, 40, 40)), (green_box, (40, 180, 40))])
    out = tracker.update(
        [PersonDetection(blue_box, 0.9), PersonDetection(green_box, 0.9)],
        frame,
    )
    ids = {t.local_track_id for t in out}
    assert len(ids) == 2
    # Both sticky ids saw a clothing change → new local ids beyond \"1\"/\"2\".
    assert any(int(i) > 2 for i in ids)
    assert "1" not in ids  # original green signature abandoned for sticky bt id 1


def test_crossing_same_color_stable_control_keeps_ids():
    """Same clothing color on both people: gate must not invent splits (stable control)."""
    tracker = RejectAndSplitPersonTracker(FPS, inner=CrossingStickyInner(), min_correlation=0.45)
    left, right = (20, 30, 40, 60), (100, 30, 40, 60)
    color = (40, 180, 40)
    for _ in range(6):
        frame = paint(200, 120, [(left, color), (right, color)])
        out = tracker.update(
            [PersonDetection(left, 0.9), PersonDetection(right, 0.9)],
            frame,
        )
        assert {t.local_track_id for t in out} == {"1", "2"}


def test_chain_track_three_appearance_changes_split_each_time():
    """One sticky inner id through three clothing changes → three local splits (chain proxy)."""
    tracker = RejectAndSplitPersonTracker(FPS, inner=StickyInner(), min_correlation=0.45)
    bbox = (20, 30, 40, 60)
    colors = [(40, 180, 40), (255, 40, 40), (40, 40, 255)]  # green, blue, red
    local_ids = []
    for color in colors:
        for _ in range(4):
            frame = paint(200, 120, [(bbox, color)])
            out = tracker.update([PersonDetection(bbox, 0.9)], frame)
            assert len(out) == 1
            local_ids.append(out[0].local_track_id)
    # Each clothing segment should use a distinct local id: 1, then 2, then 3.
    assert local_ids[0] == "1"
    assert local_ids[4] == "2"
    assert local_ids[8] == "3"
    assert set(local_ids) == {"1", "2", "3"}
