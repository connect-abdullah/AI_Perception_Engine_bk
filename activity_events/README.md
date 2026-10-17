# activity_events: activity/state + local events

Takes a finished perception run folder (`manifest.json` + `observations.jsonl` + `crops/`)
and produces per-track **activity intervals** (sitting / standing / walking / moving / still)
and **local events** (entered/left view, entered/exited zone, dwell). Output format:
[`docs/activity_event_contract.md`](../docs/activity_event_contract.md).

```
run folder ──► run.py ──► activities.jsonl + events.jsonl ──► post.py ──► backend API
                  │
                  └──► render.py ──► overlay video (check by eye / demo)
```

## Install

```
pip install -r activity_events/requirements.txt
```

## Use

```bash
# 1. (optional) draw zones: save a frame with a pixel grid, then write zones/<CAM>.json
python -m activity_events.frame VIDEO --at 10 --out frame_grid.png

# 2. activity + events for one run
python -m activity_events.run perception_runs/G330 --zones zones/G330.json --device 0

# 3. look at it
python -m activity_events.render perception_runs/G330 VIDEO --zones zones/G330.json --out g330_overlay.mp4 --start 60 --end 120

# 4. send to the backend (run observations first, then activities, then events)
python -m activity_events.post perception_runs/G330 --api http://localhost:8000
```

`--no-pose` skips the pose model (motion labels only); `--device cpu` if there is no GPU.

## Files

| File | What it does |
|---|---|
| `runfolder.py` | Loads a complete run folder (refuses `_INCOMPLETE` / unknown versions). |
| `pose.py` | Pose-model seam (`PoseEstimator`), YOLO-pose implementation, posture rule. |
| `activity.py` | Per-observation labels → smoothing → intervals with an agreement score. |
| `zones.py` | Zone polygons per camera, foot-point inside test. |
| `events.py` | View and zone events with fragment/debounce filters. |
| `run.py` | CLI: writes the result files. |
| `render.py` | Overlay video. |
| `post.py` | Uploads to the backend API. |
| `frame.py` | Grid frame for reading off zone coordinates. |

Tests: `tests/activity_events/` (synthetic run folders, fake pose model; no GPU or network).
