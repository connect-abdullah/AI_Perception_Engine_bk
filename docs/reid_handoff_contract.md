# Perception → ReID handoff contract

Status: **stable** for `contract_version` `"1"`. Field names and meanings below are
frozen for this version; a breaking change requires a new `contract_version`.
This document is the boundary between any perception producer (video → detection →
local tracking → crops) and the ReID consumer. It does not depend on any particular
producer's code. A producer is compliant if its output follows this document; the
consumer reads only this output and must not depend on, or branch on, which producer
wrote it.

### Field map (task checklist → contract names)

| Checklist name | Contract name | Where |
|---|---|---|
| `source_video_id` | `source_video_id` | manifest + every observation |
| `camera_code` | `camera_code` | **manifest only** (not duplicated on observation lines) |
| `local_track_id` | `local_track_id` | observation + track summary |
| `frame_number` | `frame_number` | observation (0-based source frames) |
| `timestamp` | **`timestamp_seconds`** | observation (float, seconds from video start) |
| `bbox` | `bbox` as `[x, y, width, height]` ints | observation |
| `confidence` | **`detection_confidence`** | observation |
| `crop_path` | `crop_path` (string or `null`) | observation |

Do not add duplicate observation fields merely to match alternate checklist names.

## Layout

One output directory per processed source video:

```
<run_dir>/
  manifest.json        # video metadata, producer metadata, per-track summary
  observations.jsonl   # one JSON object per line, one line per (track, frame) observation
  crops/<local_track_id>/<frame_number padded to 8 digits>.png
```

All paths inside the files are relative to `<run_dir>` and use `/` separators.

## Manifest (`manifest.json`)

### Must have

| Field | Type | Meaning |
|---|---|---|
| `contract_version` | string | `"1"`. Consumers reject versions they do not know. |
| `source_video_id` | string | Identifies the source video; unique across videos. |
| `camera_code` | string | Readable camera identifier, e.g. `CAM-LOBBY-01`. Same namespace as the backend `cameras.camera_code` and the ReID `ObservationContext` camera identifier. **Not** the backend camera UUID. |
| `fps` | number > 0 | The frame rate used to derive `timestamp_seconds`. |

### Should have

| Field | Type | Meaning |
|---|---|---|
| `video_start_time` | string or null | Wall-clock time of source frame 0, ISO-8601 **with timezone offset**. Null if unknown. |
| `width`, `height` | integer > 0 | Size in pixels of the decoded source frames that bboxes refer to. |
| `frame_count` | integer or null | Number of source frames, if known (may be the container's estimate). |
| `processing` | object | How the video was processed, e.g. `stride`, `crop_every`, the tracker's update rate (`fps / stride`) and the number of processed frames. |
| `timestamp_method` | string | How `timestamp_seconds` was derived, e.g. `"frame_index/fps"` (assumes constant frame rate) or `"container_pts"`. |
| `producer` | object | Free-form producer, detector and tracker metadata (names, versions, parameters). For audit only; consumers must not change behaviour based on it. |
| `num_observations` | integer | Number of lines in `observations.jsonl`. |
| `tracks` | array | Per-track summary, see below. Every value can be derived from `observations.jsonl`. |

Per-track summary entry:

| Field | Type |
|---|---|
| `local_track_id` | string |
| `first_frame_number`, `last_frame_number` | integer |
| `first_timestamp_seconds`, `last_timestamp_seconds` | number |
| `num_observations` | integer ≥ 1 |
| `num_crops` | integer ≥ 1 |

## Observations (`observations.jsonl`)

### Must have

| Field | Type | Meaning |
|---|---|---|
| `source_video_id` | string | Same value as in the manifest. |
| `local_track_id` | string | Tracker-assigned id. |
| `frame_number` | integer ≥ 0 | 0-based source-video frame number. |
| `timestamp_seconds` | number ≥ 0 | Seconds from the start of this video. |
| `bbox` | `[x, y, width, height]` integers | Box in pixels of the original source frame; `width, height > 0`. |
| `detection_confidence` | number in [0, 1] | The detector's raw score for this box. |
| `crop_path` | string or null | Relative path of the crop for this observation, or null if no crop was saved for it. |

## Rules

1. **`local_track_id` is unique only within `source_video_id`.** The same id in two videos
   says nothing about the two tracks being related.
2. **Frame numbers are 0-based source-video frame numbers.** If a producer processes only
   some frames (for example every 3rd), it keeps the original numbers (0, 3, 6, …); it does
   not renumber processed frames. Within a track, frame numbers strictly increase, and a
   track has at most one observation per frame.
3. **`timestamp_seconds` means seconds from that video's start.** Values from different
   videos must not be compared unless a shared `video_start_time` is available for both;
   the absolute time is then `video_start_time + timestamp_seconds`. Without it, the
   consumer has no cross-video timing and must treat timing context as unavailable.
4. **bbox is `[x, y, width, height]` in integer pixels of the original source frame**, with
   `(x, y)` the top-left corner, clipped to the frame. It is the detector's box matched to
   the track in that frame, not a tracker-predicted box; frames in which a track has no
   matched detection produce no observation.
5. **Crops** are the exact bbox region of the decoded source frame (no padding, no
   resizing), stored losslessly as PNG. Every track listed must have at least one crop.
6. **Crops contain no identity labels.** Nothing in the file name, path or image content
   carries an identity.
7. **No global person ID belongs in this contract.** Assigning person codes (`P001`, …) is
   the consumer's job.
8. **No identity confidence belongs in this contract.** `detection_confidence` is a detector
   score that a box contains a person; it is not evidence about who the person is, and the
   consumer must not use it as such.

## Completeness

A run directory is complete only if `manifest.json` exists and no `_INCOMPLETE` file is
present. Producers write `manifest.json` last. Consumers must not read a directory that
fails this check; it may hold partial observations and crops from a failed run.

## Data handling

Run directories contain images of real people. They must not be committed to the
repository (`perception_runs/` is gitignored for this purpose).
