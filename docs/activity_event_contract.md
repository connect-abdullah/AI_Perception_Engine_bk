# Activity / event output contract

Status: draft v0 (Abubakar). Produced by `activity_events/` from any complete run folder
that follows [`reid_handoff_contract.md`](reid_handoff_contract.md). Consumed by the
backend (`POST /api/v1/ingest/activity`, `POST /api/v1/ingest/event`) and, through the
backend, by the AI query layer.

```
<out>/                      default: <run_dir>/activity_events/
  activities.jsonl          one activity interval per line
  events.jsonl              one local event per line
  summary.json              counts + the exact settings used
```

## Keys shared by both files

| Field | Meaning |
|---|---|
| `source_video_id` | Copied from the run manifest. |
| `camera_code` | Copied from the run manifest. |
| `local_track_id` | The producer's local track id, unchanged. **Join key** with tracks/identity: `source_video_id + local_track_id`. No P-codes appear here; identity attaches through the track. |
| `method` | Rule version that produced the record (e.g. `pose_posture+foot_speed_v1`, `track_rules_v0`). |

Times: `*_seconds` fields are always present (seconds from the start of that video).
ISO-8601 `start_time` / `end_time` / `timestamp` are filled **only** when the manifest has
`video_start_time`; otherwise they are `null` and the records are not posted to the backend
(no invented times).

## `activities.jsonl`

| Field | Type | Meaning |
|---|---|---|
| `activity` | string | `sitting`, `standing`, `walking` (posture known), or `moving`, `still` (legs not visible, so only motion is known). |
| `start_seconds`, `end_seconds` | number | Interval bounds. Intervals of one track are contiguous: each ends where the next starts. |
| `start_time`, `end_time` | string or null | Absolute bounds (see Times). |
| `confidence` | number in [0, 1] | Share of the raw per-observation labels inside the interval that agree with it. An agreement score, **not** a calibrated probability. |
| `num_observations` | integer | Observations covered by the interval. |

How labels are made: posture from YOLO-pose keypoints on the track's crops (knee drop
below the hip relative to torso length), motion from foot-point speed in body-heights per
second, majority-smoothed over 2 s, intervals shorter than 1 s absorbed by a neighbour.
Tracks shorter than 1 s or with fewer than 3 observations get no intervals (fragments).

## `events.jsonl`

| `event_type` | When |
|---|---|
| `entered_view` | Track starts with its box at the frame border (needs width/height in the manifest). |
| `left_view` | Track ends with its box at the frame border. |
| `entered_zone` | Foot point crosses into a zone and stays ≥ 1 s. Not emitted if the track starts inside. |
| `exited_zone` | Foot point leaves a zone and stays out ≥ 1 s. |
| `dwell` | An inside-zone stretch lasted ≥ the zone's `dwell_seconds`. |

| Field | Type | Meaning |
|---|---|---|
| `timestamp_seconds` | number | When it happened, seconds from video start. |
| `timestamp` | string or null | Absolute time (see Times). |
| `event_meta` | object | `zone` (zone events), `frame_number`, and for `dwell`: `duration_seconds`, `end_seconds`, `ongoing_at_track_end`, `started_inside`. |
| `confidence` | null | Rule output, no score. |

Tracks that start/end away from the border produce no view events (more likely occlusion
or a tracker split). Tracks shorter than 1 s produce no events.

## Zones

`zones/<camera_code>.json` (see `zones/EXAMPLE.json`): named image-space polygons with an
optional `dwell_seconds`, drawn on a frame of size `frame_size` (rescaled automatically if
the run's frame size differs). A person is in a zone when the bottom-centre of their box
is inside the polygon. These are a stand-in until world/floor coordinates exist; the
event rules only need an inside/outside test, so swapping to floor polygons later does not
change this contract.

## Backend mapping

| File record | Endpoint | Notes |
|---|---|---|
| run observations | `POST /api/v1/ingest/run` | Sent first (chunked); tracks must exist before activities/events. |
| activity | `POST /api/v1/ingest/activity` | `source_video_id, local_track_id, activity, start_time, end_time, confidence`. |
| event | `POST /api/v1/ingest/event` | `event_type, timestamp, confidence, event_meta`; `camera_code`, `timestamp_seconds`, `method` are added to `event_meta`. |

`python -m activity_events.post RUN_DIR --api URL` does all three.

## Known limitations (v0)

- Thresholds are hand-tuned (webcam + one public pedestrian clip), not measured against
  labelled data. MEVA activity annotations would allow a proper accuracy check.
- Sitting/standing needs knees visible in the crop; far or occluded people get motion labels only.
- Crops are sparser than observations, so posture is reused for up to 1.5 s around each crop.
- Backend inserts activities/events without de-duplication: posting the same results twice stores them twice.
- Image-space zones are per camera; cross-camera "entered the building" needs the spatial layer.
