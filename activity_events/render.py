"""Overlay video for checking activity/events by eye (and for demos).

    python -m activity_events.render RUN_DIR VIDEO --out overlay.mp4
        [--results RUN_DIR/activity_events] [--zones zones/CAM.json] [--start 0] [--end 60]

Draws, per frame: each track's box and local id, its activity label at that time,
zone outlines (filled lightly while someone stands in them), and recent events as
short captions. Labels come only from the result files; nothing is recomputed here.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from activity_events.runfolder import load_run
from activity_events.zones import load_zones

COLORS = {  # BGR
    "standing": (120, 200, 80),
    "sitting": (60, 170, 255),
    "walking": (255, 140, 60),
    "moving": (255, 140, 60),
    "still": (180, 180, 180),
    None: (90, 90, 90),
}
EVENT_SHOW_SECONDS = 1.5


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _label(text, img, x, y, color):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    y = max(y, th + 6)
    cv2.rectangle(img, (x, y - th - 6), (x + tw + 6, y), color, -1)
    cv2.putText(img, text, (x + 3, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)


def render(run_dir, video_path, out_path, *, results_dir=None, zones_path=None, start=0.0, end=None) -> int:
    run = load_run(run_dir)
    results = Path(results_dir) if results_dir else run.path / "activity_events"
    intervals = defaultdict(list)
    for iv in _read_jsonl(results / "activities.jsonl"):
        intervals[iv["local_track_id"]].append(iv)
    events = _read_jsonl(results / "events.jsonl")
    zones = load_zones(zones_path, run.frame_size) if zones_path else []

    by_frame = defaultdict(list)
    for track in run.tracks.values():
        for o in track.observations:
            by_frame[o.frame_number].append(o)
    hold = max(1, int(run.manifest.get("processing", {}).get("stride", 1)))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or run.fps
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    first = int(start * fps)
    last = int(end * fps) if end is not None else None
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)

    current: dict[str, tuple[int, object]] = {}  # track -> (frame seen, observation)
    frame_no, written = first, 0
    while last is None or frame_no <= last:
        ok, img = cap.read()
        if not ok:
            break
        t = frame_no / fps
        for o in by_frame.get(frame_no, []):
            current[o.local_track_id] = (frame_no, o)
        visible = [o for f, o in current.values() if frame_no - f < hold]

        overlay = img.copy()
        for z in zones:
            pts = np.array([(int(x), int(y)) for x, y in z.polygon], dtype=np.int32)
            if any(z.contains(o.foot_point) for o in visible):
                cv2.fillPoly(overlay, [pts], (0, 200, 255))
            cv2.polylines(img, [pts], True, (0, 200, 255), 2)
            _label(z.name, img, int(pts[0][0]), int(pts[0][1]), (0, 200, 255))
        img = cv2.addWeighted(overlay, 0.18, img, 0.82, 0)

        for o in visible:
            act = next((iv["activity"] for iv in intervals.get(o.local_track_id, [])
                        if iv["start_seconds"] <= t <= iv["end_seconds"]), None)
            color = COLORS.get(act, COLORS[None])
            x, y, bw, bh = o.bbox
            cv2.rectangle(img, (x, y), (x + bw, y + bh), color, 2)
            _label(f"#{o.local_track_id} {act or '-'}", img, x, y, color)

        recent = [e for e in events if 0 <= t - e["timestamp_seconds"] <= EVENT_SHOW_SECONDS]
        for i, e in enumerate(recent[:6]):
            zone = e["event_meta"].get("zone")
            text = f"#{e['local_track_id']} {e['event_type']}" + (f" ({zone})" if zone else "")
            _label(text, img, 10, 24 + i * 22, (255, 255, 255))
        cv2.putText(img, f"{t:6.1f}s", (w - 90, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        writer.write(img)
        written += 1
        frame_no += 1
    cap.release()
    writer.release()
    return written


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("video")
    ap.add_argument("--out", required=True)
    ap.add_argument("--results", help="folder with activities.jsonl/events.jsonl (default RUN_DIR/activity_events)")
    ap.add_argument("--zones")
    ap.add_argument("--start", type=float, default=0.0, help="seconds")
    ap.add_argument("--end", type=float, default=None, help="seconds")
    args = ap.parse_args(argv)
    n = render(args.run_dir, args.video, args.out, results_dir=args.results, zones_path=args.zones,
               start=args.start, end=args.end)
    print(f"wrote {n} frames to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
