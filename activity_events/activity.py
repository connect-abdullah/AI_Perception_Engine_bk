"""Per-track activity/state intervals: sitting, standing, walking, moving, still.

How a label is chosen for each observation of a track:
  - posture (sitting / standing) comes from the pose of the nearest crop in time
    (crops are sparser than observations), or None when legs are not visible;
  - motion comes from how fast the foot point moves, in body-heights per second,
    so near and far people are treated alike;
  - sitting -> "sitting"; standing + moving -> "walking"; standing + still -> "standing";
    posture unknown -> "moving" / "still".
Labels are majority-smoothed over a time window, then merged into contiguous intervals;
intervals shorter than `min_interval_seconds` are absorbed by a neighbour.

`confidence` of an interval = share of the raw (unsmoothed) labels inside it that agree
with the interval's label. It is an agreement score, not a calibrated probability.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass

import numpy as np

from activity_events.pose import PoseEstimator, PostureSettings, posture_from_pose
from activity_events.runfolder import RunFolder, Track

LABELS = ("sitting", "standing", "walking", "moving", "still")
METHOD = "pose_posture+foot_speed_v1"


@dataclass(frozen=True)
class ActivitySettings:
    min_track_seconds: float = 1.0       # shorter tracks (tracker fragments) are skipped
    min_observations: int = 3
    posture_max_gap_seconds: float = 1.5  # how far a crop's posture may be reused in time
    speed_window_seconds: float = 1.0
    move_speed: float = 0.35              # body-heights per second; above this = moving
    smooth_window_seconds: float = 2.0
    min_interval_seconds: float = 1.0
    posture: PostureSettings = PostureSettings()


@dataclass(frozen=True)
class ActivityInterval:
    source_video_id: str
    camera_code: str
    local_track_id: str
    activity: str
    start_seconds: float
    end_seconds: float
    start_time: str | None  # ISO-8601, only when the manifest has video_start_time
    end_time: str | None
    confidence: float
    num_observations: int
    method: str = METHOD

    def to_json(self) -> dict:
        return asdict(self)

    def ingest_payload(self) -> dict:
        """Body for POST /api/v1/ingest/activity (requires known absolute times)."""
        if self.start_time is None:
            raise ValueError("start_time unknown (manifest has no video_start_time); refusing to invent one")
        return {
            "source_video_id": self.source_video_id,
            "local_track_id": self.local_track_id,
            "activity": self.activity,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "confidence": round(self.confidence, 4),
        }


# --------------------------------------------------------------------------- per-observation signals


def track_postures(run: RunFolder, track: Track, estimator: PoseEstimator | None, settings: ActivitySettings):
    """[(timestamp_seconds, 'sitting'|'standing')] for crops where posture could be determined."""
    if estimator is None:
        return []
    samples = []
    for obs in track.observations:
        crop = run.crop_file(obs)
        if crop is None or not crop.is_file():
            continue
        posture = posture_from_pose(estimator.estimate(crop), settings.posture)
        if posture is not None:
            samples.append((obs.timestamp_seconds, posture))
    return samples


def _posture_at(t: float, samples, max_gap: float) -> str | None:
    best, best_dt = None, max_gap
    for ts, posture in samples:
        dt = abs(ts - t)
        if dt <= best_dt:
            best, best_dt = posture, dt
    return best


def foot_speeds(track: Track, window: float) -> list[float | None]:
    """Foot-point speed in body-heights/second around each observation (None if too little data)."""
    obs = track.observations
    times = np.array([o.timestamp_seconds for o in obs])
    feet = np.array([o.foot_point for o in obs])
    heights = np.array([max(o.bbox[3], 1) for o in obs], dtype=float)
    speeds: list[float | None] = []
    for i, t in enumerate(times):
        idx = np.where(np.abs(times - t) <= window / 2.0)[0]
        a, b = idx.min(), idx.max()
        dt = times[b] - times[a]
        if dt < 0.2:
            speeds.append(None)
            continue
        dist = float(np.linalg.norm(feet[b] - feet[a]))
        speeds.append(dist / float(np.median(heights[a : b + 1])) / dt)
    return speeds


def raw_label(posture: str | None, speed: float | None, move_speed: float) -> str:
    moving = speed is not None and speed > move_speed
    if posture == "sitting":
        return "sitting"
    if posture == "standing":
        return "walking" if moving else "standing"
    return "moving" if moving else "still"


def _smooth(times: list[float], labels: list[str], window: float) -> list[str]:
    out = []
    for t, own in zip(times, labels):
        votes = Counter(lab for ts, lab in zip(times, labels) if abs(ts - t) <= window / 2.0)
        top = max(votes.values())
        # tie-break: prefer the observation's own label, then a fixed order
        out.append(own if votes[own] == top else next(l for l in LABELS if votes.get(l) == top))
    return out


# --------------------------------------------------------------------------- intervals


def _segments(times: list[float], labels: list[str]) -> list[list]:
    """[[label, start_idx, end_idx_exclusive], ...] runs of equal labels."""
    segs = []
    for i, lab in enumerate(labels):
        if segs and segs[-1][0] == lab:
            segs[-1][2] = i + 1
        else:
            segs.append([lab, i, i + 1])
    return segs


def _seg_duration(seg, times) -> float:
    lab, a, b = seg
    end = times[b] if b < len(times) else times[-1]
    return end - times[a]


def _absorb_short(segs, times, min_len: float):
    changed = True
    while changed and len(segs) > 1:
        changed = False
        for k, seg in enumerate(segs):
            if _seg_duration(seg, times) >= min_len:
                continue
            # merge into the longer neighbour
            left = segs[k - 1] if k > 0 else None
            right = segs[k + 1] if k + 1 < len(segs) else None
            target = left if right is None or (left is not None and _seg_duration(left, times) >= _seg_duration(right, times)) else right
            if target is left:
                left[2] = seg[2]
            else:
                right[1] = seg[1]
            del segs[k]
            # re-join equal neighbours
            i = 0
            while i + 1 < len(segs):
                if segs[i][0] == segs[i + 1][0]:
                    segs[i][2] = segs[i + 1][2]
                    del segs[i + 1]
                else:
                    i += 1
            changed = True
            break
    return segs


def track_intervals(
    run: RunFolder,
    track: Track,
    estimator: PoseEstimator | None,
    settings: ActivitySettings = ActivitySettings(),
) -> list[ActivityInterval]:
    obs = track.observations
    if len(obs) < settings.min_observations or track.duration_seconds < settings.min_track_seconds:
        return []
    times = [o.timestamp_seconds for o in obs]
    postures = track_postures(run, track, estimator, settings)
    speeds = foot_speeds(track, settings.speed_window_seconds)
    raw = [
        raw_label(_posture_at(t, postures, settings.posture_max_gap_seconds), s, settings.move_speed)
        for t, s in zip(times, speeds)
    ]
    smooth = _smooth(times, raw, settings.smooth_window_seconds)
    segs = _absorb_short(_segments(times, smooth), times, settings.min_interval_seconds)

    out = []
    for lab, a, b in segs:
        start = times[a]
        end = times[b] if b < len(times) else times[-1]  # contiguous: ends where the next begins
        agree = sum(1 for r in raw[a:b] if r == lab) / (b - a)
        out.append(
            ActivityInterval(
                source_video_id=run.source_video_id,
                camera_code=run.camera_code,
                local_track_id=track.local_track_id,
                activity=lab,
                start_seconds=round(start, 3),
                end_seconds=round(end, 3),
                start_time=run.absolute_time(start),
                end_time=run.absolute_time(end),
                confidence=round(agree, 4),
                num_observations=b - a,
            )
        )
    return out


def run_activity(run: RunFolder, estimator: PoseEstimator | None, settings: ActivitySettings = ActivitySettings()):
    """(intervals, summary) for every track in the run."""
    intervals: list[ActivityInterval] = []
    skipped = 0
    for tid in sorted(run.tracks, key=lambda s: (0, int(s), "") if s.isdigit() else (1, 0, s)):
        ivs = track_intervals(run, run.tracks[tid], estimator, settings)
        if not ivs:
            skipped += 1
        intervals.extend(ivs)
    summary = {
        "tracks_total": len(run.tracks),
        "tracks_labelled": len(run.tracks) - skipped,
        "tracks_skipped_short": skipped,
        "intervals": len(intervals),
        "by_activity": dict(Counter(i.activity for i in intervals)),
        "pose_model": getattr(estimator, "name", None),
        "absolute_times": run.video_start_time is not None,
    }
    return intervals, summary
