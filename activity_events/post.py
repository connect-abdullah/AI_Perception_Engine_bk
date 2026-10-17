"""Send a run's observations, activities and events to the backend API.

    python -m activity_events.post RUN_DIR --api http://localhost:8000
        [--results RUN_DIR/activity_events] [--skip-run] [--chunk 2000]

Order matters: the backend resolves activities/events by source_video_id + local_track_id,
so the run's observations go first (POST /api/v1/ingest/run, idempotent upsert, sent in
chunks), then each activity (POST /api/v1/ingest/activity) and event (POST /api/v1/ingest/event).

Refuses to send anything when the manifest has no video_start_time: the backend needs
absolute times, and inventing them would put fake times in the database.

Note: the backend currently inserts activities/events without de-duplication, so running
this twice for the same results stores them twice.
"""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

from activity_events.runfolder import RunFolder, load_run

Send = Callable[[str, dict], dict]


def http_sender(api: str, timeout: float = 120.0) -> Send:
    base = api.rstrip("/")

    def send(path: str, body: dict) -> dict:
        req = urllib.request.Request(
            base + path, data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read() or b"{}")
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"POST {path} -> {e.code}: {e.read().decode('utf-8', 'replace')[:500]}") from None

    return send


def run_payloads(run: RunFolder, chunk: int = 2000):
    """Bodies for POST /api/v1/ingest/run, `chunk` observations each."""
    rows = [o for t in run.tracks.values() for o in t.observations]
    rows.sort(key=lambda o: (o.frame_number, o.local_track_id))
    size = run.frame_size
    for i in range(0, max(len(rows), 1), chunk):
        yield {
            "source_video_id": run.source_video_id,
            "camera_code": run.camera_code,
            "fps": run.fps,
            "video_start_time": run.manifest.get("video_start_time"),
            "width": size[0] if size else None,
            "height": size[1] if size else None,
            "observations": [
                {"local_track_id": o.local_track_id, "frame_number": o.frame_number,
                 "timestamp_seconds": o.timestamp_seconds, "bbox": list(o.bbox),
                 "detection_confidence": o.detection_confidence, "crop_path": o.crop_path}
                for o in rows[i : i + chunk]
            ],
        }


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def post_results(run_dir, send: Send, *, results_dir=None, skip_run=False, chunk=2000) -> dict:
    from activity_events.activity import ActivityInterval
    from activity_events.events import LocalEvent

    run = load_run(run_dir)
    if run.video_start_time is None:
        raise ValueError("manifest has no video_start_time: refusing to post (the backend would store invented times)")
    results = Path(results_dir) if results_dir else run.path / "activity_events"
    counts = {"run_chunks": 0, "activities": 0, "events": 0}

    if not skip_run:
        for body in run_payloads(run, chunk):
            send("/api/v1/ingest/run", body)
            counts["run_chunks"] += 1
    for row in _read_jsonl(results / "activities.jsonl"):
        send("/api/v1/ingest/activity", ActivityInterval(**row).ingest_payload())
        counts["activities"] += 1
    for row in _read_jsonl(results / "events.jsonl"):
        send("/api/v1/ingest/event", LocalEvent(**row).ingest_payload())
        counts["events"] += 1
    return counts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--api", required=True, help="backend base URL, e.g. http://localhost:8000")
    ap.add_argument("--results", help="default: RUN_DIR/activity_events")
    ap.add_argument("--skip-run", action="store_true", help="observations were already ingested")
    ap.add_argument("--chunk", type=int, default=2000)
    args = ap.parse_args(argv)
    counts = post_results(args.run_dir, http_sender(args.api), results_dir=args.results,
                          skip_run=args.skip_run, chunk=args.chunk)
    print(json.dumps(counts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
