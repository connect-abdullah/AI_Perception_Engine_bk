"""Activity + events for one perception run folder.

    python -m activity_events.run RUN_DIR [--zones zones/CAM.json] [--out OUT_DIR]
        [--no-pose] [--pose-model yolo11n-pose.pt] [--device cpu|0]

Writes to OUT_DIR (default: RUN_DIR/activity_events/):
    activities.jsonl   one ActivityInterval per line
    events.jsonl       one LocalEvent per line
    summary.json       counts + the settings used

--no-pose skips the pose model: only motion labels (moving / still) are produced.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from activity_events.activity import ActivitySettings, run_activity
from activity_events.events import EventSettings, run_events
from activity_events.runfolder import load_run
from activity_events.zones import load_zones


def write_jsonl(path: Path, records) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.to_json(), ensure_ascii=False) + "\n")
    tmp.replace(path)


def process(run_dir, *, out_dir=None, zones_path=None, estimator=None,
            activity_settings=ActivitySettings(), event_settings=EventSettings()) -> dict:
    run = load_run(run_dir)
    out = Path(out_dir) if out_dir else run.path / "activity_events"
    out.mkdir(parents=True, exist_ok=True)
    zones = load_zones(zones_path, run.frame_size) if zones_path else []

    intervals, a_summary = run_activity(run, estimator, activity_settings)
    events, e_summary = run_events(run, zones, event_settings)
    write_jsonl(out / "activities.jsonl", intervals)
    write_jsonl(out / "events.jsonl", events)
    summary = {
        "run_dir": str(run.path),
        "source_video_id": run.source_video_id,
        "camera_code": run.camera_code,
        "activity": a_summary,
        "events": e_summary,
        "settings": {"activity": asdict(activity_settings), "events": asdict(event_settings)},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--zones", help="zones JSON for this camera (optional)")
    ap.add_argument("--out", help="output directory (default: RUN_DIR/activity_events)")
    ap.add_argument("--no-pose", action="store_true", help="motion only, no pose model")
    ap.add_argument("--pose-model", default="yolo11n-pose.pt")
    ap.add_argument("--device", default=None, help="e.g. cpu or 0 for the first GPU")
    args = ap.parse_args(argv)

    estimator = None
    if not args.no_pose:
        from activity_events.pose import YoloPoseEstimator

        estimator = YoloPoseEstimator(args.pose_model, device=args.device)
    summary = process(args.run_dir, out_dir=args.out, zones_path=args.zones, estimator=estimator)
    print(json.dumps({k: summary[k] for k in ("source_video_id", "camera_code", "activity", "events")}, indent=2))
    if not summary["activity"]["absolute_times"]:
        print("note: manifest has no video_start_time, so records have offsets only "
              "(start_seconds) and cannot be posted to the backend.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
