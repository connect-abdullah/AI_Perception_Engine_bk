# Architecture Overview

A short, plain-language tour of the AI Perception Engine as it stands in this branch.
For the exact file format between perception and ReID, see
[`reid_handoff_contract.md`](reid_handoff_contract.md).

---

## 1. Project Goal

The system should turn recorded camera video into structured, searchable records:

- **person detections**: where a person is in each frame
- **local tracks**: the same person followed across nearby frames of one video
- **global anonymous identities** such as `P001`: the same person recognised across videos and cameras, without knowing their name
- **time, location and activity data**: when and where each person was seen, and what they were doing
- **structured records** that a backend database and an AI query layer can use

Not all of this exists yet. Section 9 lists what is built and what is not.

---

## 2. Top-Level Folder Map

| Folder | What it is |
|---|---|
| `app/` | Backend side: FastAPI app and the database models (cameras, rooms, videos, tracks, observations, persons, events). Owned by the backend teammate. |
| `alembic/`, `sql/`, `scripts/` | Database migrations, the SQL schema, and seed/check scripts for the backend. |
| `perception_minimal/` | Video in → people found → people followed → files out. A temporary, minimal version of Person 2's work, built so ReID can be integrated; meant to be replaceable. |
| `reid/` | Person re-identification: turns person crops into embeddings and decides which anonymous global identity (`P001`, …) a track belongs to. |
| `tests/` | Automated tests. `tests/perception_minimal/` and `tests/reid/` cover the two pipelines; `tests/test_metadata.py` covers the database models. |
| `docs/` | Contracts and architecture notes (this file, the handoff contract). |

The three parts do not import each other:

```
app/  <-- (not connected yet) --  reid/  <-- (files on disk only) --  perception_minimal/
```

---

## 3. Perception Pipeline (`perception_minimal/`)

```
video file
   │
   ▼
video.py ──► frame + frame_number + timestamp_seconds
   │
   ▼
detector.py ──► people in this frame: box + detector score
   │
   ▼
tracker.py ──► the same boxes, each with a local track id ("7")
   │
   ▼
export.py ──► manifest.json, observations.jsonl, crops/
```

| File | What it does |
|---|---|
| `video.py` | Reads a video file frame by frame, giving each frame its original frame number and its time in seconds from the start of the video. |
| `detector.py` | Finds people in a frame and returns a box and a detector score for each (uses a pretrained Faster R-CNN; development use only, see section 9). |
| `tracker.py` | Links detections across nearby frames so the same person keeps the same local track id (uses ByteTrack). |
| `export.py` | Writes the results to disk: one line per detection, a summary per track, and cropped images of each person. |
| `run.py` | The command you run (`python -m perception_minimal.run VIDEO ...`) to connect all of the above for one video. |

**A local track id is only meaningful inside one video run.** `TRACK_7` in video A and
`TRACK_7` in video B are unrelated. If a person leaves the view and comes back, they may
get a new local id. Working out who is who across videos is ReID's job, not the tracker's.

---

## 4. Perception → ReID Handoff

Perception and ReID only meet through files on disk. Each processed video produces one folder:

```
<run folder>/
  manifest.json          # about the video and the run, plus one summary per track
  observations.jsonl     # one line per person-detection that belongs to a track
  crops/<local_track_id>/<frame_number>.png   # the cropped person images
```

Important fields:

| Field | Meaning |
|---|---|
| `source_video_id` | Which video this came from (by default a fingerprint of the video file). |
| `camera_code` | Readable camera name such as `CAM-LOBBY-01` (not a database id). |
| `local_track_id` | The tracker's id; only unique within this one video. |
| `frame_number` | The original frame number in the video, starting at 0. |
| `timestamp_seconds` | Seconds from the start of this video. |
| `bbox` | The person's box `[x, y, width, height]` in pixels. |
| `detection_confidence` | The detector's score that this box contains a person. It says nothing about who the person is. |
| `crop_path` | Where the cropped image for this detection is, if one was saved. |

**This handoff deliberately contains no global identity.** There is no `P001` and no
"same person as" information in it. Any team member's perception code can produce these
files, and ReID does not care which one did.

A run folder is only complete if `manifest.json` exists and there is no `_INCOMPLETE` file.

---

## 5. ReID Pipeline (`reid/`)

```
crops of one local track
   │
   ▼
embedder.py ──► one embedding (a list of numbers) per crop
   │
   ▼
tracklet.py ──► one combined embedding for the whole track
   │
   ▼
identity.py ──► known identities ranked by similarity: P001 0.81, P004 0.62, ...
   │
   ▼
decision.py ──► visual decision: probable_match / uncertain / new
   │
   │            context.py ──► was the camera/time move plausible / impossible / unknown?
   │                │
   ▼                ▼
fusion.py ──► final status
```

| File | What it does |
|---|---|
| `embedder.py` | Turns a person crop into an embedding: a list of numbers describing appearance (uses a pretrained OSNet model). |
| `_osnet.py` | The OSNet network definition used by `embedder.py`. |
| `similarity.py` | Measures how alike two embeddings are (cosine similarity; a score, not a probability). |
| `tracklet.py` | Combines the embeddings of several crops from one local track into one embedding. |
| `identity.py` | Keeps the list of anonymous global identities (`P001`, `P002`, …) and ranks them against a new track. |
| `decision.py` | Looks at the ranking and says `probable_match`, `uncertain` or `new`, using thresholds the caller must supply. |
| `context.py` | Checks whether moving between two cameras in the observed time is plausible, impossible or unknown, using camera rules the caller supplies. |
| `fusion.py` | Combines the visual decision with the context result into a final status. |
| `compare.py`, `evaluate.py` | Developer tools: compare two crops, or measure matching quality on a labelled folder of crops. |

### `context.py` is not `fusion.py`

- **`context.py`** asks: *"Is this movement physically plausible?"*
  For example: last seen on `CAM_A` at 10:00:00, now on `CAM_B` at 10:00:01, but the
  walk takes at least 30 seconds → **impossible**. It never looks at appearance.
- **`fusion.py`** asks: *"Given the visual decision and the context result, what final status should we keep?"*
  It is deliberately conservative:
  - context can only **veto**: `probable_match` + **impossible** → `uncertain`
  - plausible context never upgrades a weak visual result
  - unknown context changes nothing
  - `new` stays `new`

None of these modules creates or updates identities by itself. Whoever runs the pipeline
decides what to do with the final status (for example: create `P001` on `new`, attach the
track on `probable_match`, leave it unassigned on `uncertain`).

---

## 6. Example Person Journey

**1. First sighting**

```
VIDEO_A (CAM_LOBBY), TRACK_7
  → 6 crops → embeddings → one track embedding
  → no known identities yet
  → decision: NEW
  → create P001 from TRACK_7
```

**2. Seen again, plausible move**

```
VIDEO_B (CAM_CORRIDOR), TRACK_3
  → track embedding is clearly closest to P001
  → decision: PROBABLE_MATCH P001
  → context: last seen CAM_LOBBY 10:02:10, now CAM_CORRIDOR 10:03:00, rule says 20–300 s → PLAUSIBLE
  → fusion: PROBABLE_MATCH P001 (kept)
  → TRACK_3 is attached to P001
```

**3. Looks the same, but the move is impossible**

```
VIDEO_C (CAM_ROOF), TRACK_12
  → track embedding is clearly closest to P001
  → decision: PROBABLE_MATCH P001
  → context: last seen CAM_CORRIDOR 10:03:00, now CAM_ROOF 10:03:05, rule says at least 120 s → IMPOSSIBLE
  → fusion: UNCERTAIN (vetoed by context)
  → TRACK_12 is not attached to anyone; kept for review
```

The camera names and times above are examples. Real transition times have not been measured yet.

---

## 7. Local ID vs Global ID

| | `TRACK_7` (local track id) | `P001` (global identity) |
|---|---|---|
| Created by | the tracker (`perception_minimal/tracker.py`) | the ReID pipeline (`reid/identity.py`) |
| Lives for | one video / one run | across all videos and cameras |
| Means | "the same box followed across nearby frames" | "the same anonymous person seen again" |
| Same value elsewhere | unrelated (`TRACK_7` in another video is someone else, or no one) | the same person |
| Contains a name? | no | no; it is anonymous (the backend has an optional `known_name`, not used by ReID) |

---

## 8. Where Person 3 (Backend) Fits

```
global identity result (P001, status, reason)
   +
track observations (camera, time, box) and, later, activities/events
   │
   ▼
backend / database   (app/: videos, tracks, observations, persons, events)
   │
   ▼
AI query layer       (not built yet)
```

The database models in `app/` already have places for this data (`tracks.person_id`,
`observations`, `persons.person_code`, `events`). HTTP ingest routes can write
perception observations and activity/event payloads into the database, but there is
**no automatic disk→DB bridge** from perception run folders, **no ReID file consumer**
of the handoff contract, and **the AI query layer does not exist yet**.

Example questions this should eventually answer:

- Who entered Room 3 between 10am and 1pm?
- Where was P001 last seen?
- Who was in the lobby?
- Show P001's movement history.

---

## 9. Current Status

### Implemented technically (code exists and has automated tests)

- video ingestion (frame numbers, timestamps, frame skipping)
- person detection (pretrained Faster R-CNN, person class only)
- local tracking (ByteTrack + experimental HSV reject-and-split gate)
- structured output (manifest, observations, crops; handoff contract v1 stable)
- HTTP ingest for observations / activity / events (sink ready; event producer not wired)
- crop embeddings (pretrained OSNet)
- tracklet aggregation
- anonymous identity registry and candidate ranking
- decision semantics (probable_match / uncertain / new)
- camera/time context checks
- conservative visual + context fusion

The tests use synthetic images, synthetic video and stand-in detectors. They show the code
behaves as designed, not that it recognises real people well.

### Not yet validated / not yet complete

- **Real-person ReID accuracy**: not measured on real, consented footage.
- **Real cross-camera performance**: not measured.
- **Calibrated thresholds**: the decision thresholds, the detector score threshold and the tracker settings have not been tuned on real data.
- **Real camera transition timings**: no measured walking times between cameras.
- **Final production model licensing**: the current detector (COCO) and ReID (MSMT17-trained OSNet) weights are for development only and are not cleared for commercial use.
- **End-to-end Person 2 → ReID consumer**: nothing yet reads the handoff files and runs ReID on them.
- **Backend integration**: ingest API exists; automatic write from perception run dirs is not.
- **Activity recognition and entry/exit events**: detection producer not implemented (ingest sink exists).
- **AI query validation**: no query layer exists.
- **Persistence of identities**: the identity registry lives in memory only.
- **Optional face / gait recognition**: not implemented.

---

## 10. Future Extensions

- **Face recognition** could later be added as an extra source of identity evidence when a face is clearly visible.
- **Gait recognition** (how someone walks) could later be added as another evidence source.
- Neither is implemented now. **Appearance-based ReID remains the current primary identity mechanism.**
- A new evidence source would most likely sit next to `context.py` and feed the fusion step, rather than replace the existing pieces. That design has not been worked out yet.
