"""Helpers to write small synthetic run folders and a fake pose estimator."""

import json
from pathlib import Path

import numpy as np
import pytest

from activity_events.pose import Pose

W, H = 640, 480


def write_run(root: Path, tracks: dict, *, fps=10.0, start_time="2026-10-17T10:00:00+00:00",
              size=(W, H), crops=True, source_video_id="vid-1", camera_code="CAM-A") -> Path:
    """tracks: {track_id: [(frame_number, (x, y, w, h)), ...]}; writes a complete run folder."""
    root.mkdir(parents=True, exist_ok=True)
    lines = []
    summaries = []
    for tid, obs in tracks.items():
        for frame, bbox in obs:
            crop_path = None
            if crops:
                crop_path = f"crops/{tid}/{frame:08d}.png"
                p = root / crop_path
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"png")  # content is never decoded by the fake estimator
            lines.append({
                "source_video_id": source_video_id, "local_track_id": str(tid), "frame_number": frame,
                "timestamp_seconds": frame / fps, "bbox": list(bbox), "detection_confidence": 0.9,
                "crop_path": crop_path,
            })
        summaries.append({"local_track_id": str(tid), "num_observations": len(obs)})
    (root / "observations.jsonl").write_text("".join(json.dumps(l) + "\n" for l in lines), encoding="utf-8")
    manifest = {"contract_version": "1", "source_video_id": source_video_id, "camera_code": camera_code,
                "fps": fps, "video_start_time": start_time, "tracks": summaries}
    if size:
        manifest["width"], manifest["height"] = size
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def pose(posture: str | None) -> Pose | None:
    """A COCO-17 pose that the posture rule reads as `posture` ('sitting'/'standing'), or None."""
    if posture is None:
        return None
    kps = np.zeros((17, 2), dtype=np.float32)
    conf = np.full(17, 0.9, dtype=np.float32)
    knee_drop = 50.0 if posture == "standing" else 10.0  # torso length is 50
    for sho, hip, knee, x in ((5, 11, 13, 40.0), (6, 12, 14, 60.0)):
        kps[sho] = (x, 20.0)
        kps[hip] = (x, 70.0)
        kps[knee] = (x, 70.0 + knee_drop)
    return Pose(kps, conf)


class FakePose:
    """Returns a fixed posture per (track, frame) taken from the crop path, default for others."""

    name = "fake"

    def __init__(self, by_frame=None, default="standing"):
        self.by_frame = by_frame or {}
        self.default = default
        self.calls = 0

    def estimate(self, crop_path):
        self.calls += 1
        frame = int(Path(crop_path).stem)
        return pose(self.by_frame.get(frame, self.default))


@pytest.fixture
def make_run(tmp_path):
    def _make(tracks, **kw):
        return write_run(tmp_path / "run", tracks, **kw)
    return _make
