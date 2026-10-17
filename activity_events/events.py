"""Local, per-camera events derived from tracks (and optionally zones).

Event types (strings, stored as backend events.event_type):
  entered_view   track starts at the frame border (someone walked into view)
  left_view      track ends at the frame border (someone walked out of view)
  entered_zone   foot point moves from outside to inside a zone and stays there
  exited_zone    foot point moves from inside to outside a zone and stays out
  dwell          an inside-zone stretch lasted at least the zone's dwell time

Deliberately conservative, because local tracks fragment (the tracker splits rather
than merges):
  - tracks shorter than `min_track_seconds` produce no events;
  - a track that starts/ends away from the frame border produces no entered/left_view:
    that is more likely occlusion or a tracker split than a real entrance/exit;
  - a zone change only counts if the new side is held for `min_zone_seconds`;
  - a track that is already inside a zone at its first observation gets no entered_zone.

All events are local to one video/camera and one local track. Who the person is
(P-codes) is added later by the identity layer through the track link.
`confidence` is null: these are rule outputs, not model scores.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field

from activity_events.runfolder import RunFolder, Track
from activity_events.zones import Zone

METHOD = "track_rules_v0"
# Order for events that share a timestamp.
_ORDER = {"entered_view": 0, "entered_zone": 1, "dwell": 2, "exited_zone": 3, "left_view": 4}


@dataclass(frozen=True)
class EventSettings:
    min_track_seconds: float = 1.0
    min_observations: int = 3
    edge_margin_px: int = 12      # box within this many pixels of the border counts as "at the edge"
    min_zone_seconds: float = 1.0
    default_dwell_seconds: float | None = None  # used for zones without their own dwell_seconds


@dataclass(frozen=True)
class LocalEvent:
    source_video_id: str
    camera_code: str
    local_track_id: str
    event_type: str
    timestamp_seconds: float
    timestamp: str | None  # ISO-8601, only when the manifest has video_start_time
    event_meta: dict = field(default_factory=dict)
    confidence: float | None = None
    method: str = METHOD

    def to_json(self) -> dict:
        return asdict(self)

    def ingest_payload(self) -> dict:
        """Body for POST /api/v1/ingest/event (requires a known absolute time)."""
        if self.timestamp is None:
            raise ValueError("timestamp unknown (manifest has no video_start_time); refusing to invent one")
        meta = dict(self.event_meta)
        meta.update(camera_code=self.camera_code, timestamp_seconds=self.timestamp_seconds, method=self.method)
        return {
            "source_video_id": self.source_video_id,
            "local_track_id": self.local_track_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            "confidence": self.confidence,
            "event_meta": meta,
        }


def _at_edge(bbox, frame_size, margin) -> bool:
    x, y, w, h = bbox
    fw, fh = frame_size
    return x <= margin or y <= margin or x + w >= fw - margin or y + h >= fh - margin


def _stable_runs(times: list[float], flags: list[bool], min_len: float) -> list[list]:
    """[[flag, start_idx, end_idx_exclusive]] with runs shorter than min_len absorbed by a neighbour."""
    runs: list[list] = []
    for i, f in enumerate(flags):
        if runs and runs[-1][0] == f:
            runs[-1][2] = i + 1
        else:
            runs.append([f, i, i + 1])

    def dur(r):
        return (times[r[2]] if r[2] < len(times) else times[-1]) - times[r[1]]

    # Absorb short runs (a blip, or a short tail/head) into a neighbour until all are long
    # enough, then re-join equal neighbours. A short first run joins the next one.
    while len(runs) > 1:
        k = next((i for i, r in enumerate(runs) if dur(r) < min_len), None)
        if k is None:
            break
        if k == 0:
            runs[1][1] = runs[0][1]
        else:
            runs[k - 1][2] = runs[k][2]
        del runs[k]
        i = 0
        while i + 1 < len(runs):
            if runs[i][0] == runs[i + 1][0]:
                runs[i][2] = runs[i + 1][2]
                del runs[i + 1]
            else:
                i += 1
    return runs


def track_events(
    run: RunFolder,
    track: Track,
    zones: list[Zone],
    settings: EventSettings = EventSettings(),
) -> list[LocalEvent]:
    obs = track.observations
    if len(obs) < settings.min_observations or track.duration_seconds < settings.min_track_seconds:
        return []

    def make(event_type, t, **meta):
        return LocalEvent(
            source_video_id=run.source_video_id,
            camera_code=run.camera_code,
            local_track_id=track.local_track_id,
            event_type=event_type,
            timestamp_seconds=round(t, 3),
            timestamp=run.absolute_time(t),
            event_meta=meta,
        )

    events: list[LocalEvent] = []
    size = run.frame_size
    if size is not None:
        if _at_edge(obs[0].bbox, size, settings.edge_margin_px):
            events.append(make("entered_view", obs[0].timestamp_seconds, frame_number=obs[0].frame_number))
        if _at_edge(obs[-1].bbox, size, settings.edge_margin_px):
            events.append(make("left_view", obs[-1].timestamp_seconds, frame_number=obs[-1].frame_number))

    times = [o.timestamp_seconds for o in obs]
    for zone in zones:
        flags = [zone.contains(o.foot_point) for o in obs]
        runs = _stable_runs(times, flags, settings.min_zone_seconds)
        for k, (inside, a, b) in enumerate(runs):
            start = times[a]
            if k > 0:
                events.append(make("entered_zone" if inside else "exited_zone", start, zone=zone.name,
                                   frame_number=obs[a].frame_number))
            dwell = zone.dwell_seconds if zone.dwell_seconds is not None else settings.default_dwell_seconds
            if inside and dwell is not None:
                end = times[b] if b < len(times) else times[-1]
                if end - start >= dwell:
                    events.append(make("dwell", start, zone=zone.name, duration_seconds=round(end - start, 3),
                                       end_seconds=round(end, 3), ongoing_at_track_end=b >= len(times),
                                       started_inside=k == 0))
    events.sort(key=lambda e: (e.timestamp_seconds, _ORDER.get(e.event_type, 9)))
    return events


def run_events(run: RunFolder, zones: list[Zone], settings: EventSettings = EventSettings()):
    """(events, summary) for every track in the run."""
    events: list[LocalEvent] = []
    for tid in sorted(run.tracks, key=lambda s: (0, int(s), "") if s.isdigit() else (1, 0, s)):
        events.extend(track_events(run, run.tracks[tid], zones, settings))
    events.sort(key=lambda e: (e.timestamp_seconds, e.local_track_id, _ORDER.get(e.event_type, 9)))
    summary = {
        "tracks_total": len(run.tracks),
        "events": len(events),
        "by_type": dict(Counter(e.event_type for e in events)),
        "zones": [z.name for z in zones],
        "edge_events": run.frame_size is not None,
        "absolute_times": run.video_start_time is not None,
    }
    return events, summary
