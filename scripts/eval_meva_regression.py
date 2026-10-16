"""Evaluate the MEVA track-stability regression set against a perception run.

Usage:
    python scripts/eval_meva_regression.py \\
        --g330 eval_data/meva/baseline/G330_observations.jsonl \\
        --g299 eval_data/meva/baseline/G299_observations.jsonl \\
        --label baseline

This does NOT claim a global tracker accuracy number. It reports whether each
named failure / stable-control track from docs/regression/ is present in the
run and summarises its frame span / observation count. Visual identity
continuity (switch vs stable) still requires crop-strip review when crops
are available.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "docs" / "regression" / "track_stability_observations.csv"

PRIORITY = {
    "crossing": [
        ("G330", "864"),
        ("G299", "808"),
        ("G299", "580"),
        ("G330", "307"),
        ("G299", "293"),
    ],
    "chain": [
        ("G330", "518"),
        ("G330", "539"),
        ("G330", "769"),
        ("G330", "375"),
        ("G330", "118"),
        ("G299", "686"),
        ("G299", "35"),
    ],
    "wrong_first": [("G330", "111"), ("G299", "748")],
    "fragment": [("G299", "360"), ("G299", "412")],
    "stable": [
        ("G330", "73"),
        ("G299", "119"),
        ("G299", "156"),
        ("G299", "412"),
        ("G299", "644"),
    ],
    "short": [
        ("G330", "195"),
        ("G330", "757"),
        ("G330", "42"),
        ("G330", "873"),
        ("G299", "192"),
        ("G330", "824"),
    ],
}


def load_obs(path: Path) -> dict[str, list[dict]]:
    by_track: dict[str, list[dict]] = defaultdict(list)
    with path.open(encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            by_track[str(o["local_track_id"])].append(o)
    for tid in by_track:
        by_track[tid].sort(key=lambda x: x["frame_number"])
    return by_track


def find_track(tracks: dict[str, list[dict]], tid: str) -> list[dict] | None:
    if tid in tracks:
        return tracks[tid]
    try:
        n = str(int(tid))
    except ValueError:
        return None
    for k in tracks:
        try:
            if str(int(k)) == n:
                return tracks[k]
        except ValueError:
            continue
    return None


def summarise(seq: list[dict]) -> dict:
    frames = [o["frame_number"] for o in seq]
    confs = [float(o["detection_confidence"]) for o in seq]
    return {
        "n": len(seq),
        "first": frames[0],
        "last": frames[-1],
        "span_s": (frames[-1] - frames[0]) / 30.0,
        "mean_conf": sum(confs) / len(confs),
        "crops": sum(1 for o in seq if o.get("crop_path")),
    }


def csv_class(camera: str, tid: str) -> str:
    with CSV_PATH.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["camera"] == camera and row["track"] == tid:
                return row["class"]
    return "?"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--g330", type=Path, required=True)
    p.add_argument("--g299", type=Path, required=True)
    p.add_argument("--label", default="run")
    args = p.parse_args()

    data = {
        "G330": load_obs(args.g330),
        "G299": load_obs(args.g299),
    }
    print(f"=== MEVA regression eval: {args.label} ===")
    for cam, tracks in data.items():
        print(
            f"{cam}: {sum(len(v) for v in tracks.values())} obs, "
            f"{len(tracks)} tracks  ({args.__dict__[cam.lower()]})"
        )

    print()
    print(
        f"{'CASE':12} {'CAM':5} {'TID':6} {'CSV_CLASS':28} "
        f"{'STATUS':8} {'n':>4} {'frames':>14} {'span_s':>7}"
    )
    print("-" * 95)

    missing = 0
    found = 0
    for kind, cases in PRIORITY.items():
        for cam, tid in cases:
            seq = find_track(data[cam], tid)
            cls = csv_class(cam, tid)
            if seq is None:
                missing += 1
                print(f"{kind:12} {cam:5} T{tid:5} {cls:28} MISSING")
                continue
            found += 1
            s = summarise(seq)
            print(
                f"{kind:12} {cam:5} T{tid:5} {cls:28} FOUND    "
                f"{s['n']:4} {s['first']}-{s['last']:<8} {s['span_s']:7.1f}"
            )

    print()
    print(f"Present: {found}  Missing: {missing}")
    print()
    print(
        "NOTE: FOUND means the local track id exists with the expected frame "
        "window. Switch / chain / fragmentation verdicts still need crop-strip "
        "or visual review; do not treat this table as a global accuracy score."
    )


if __name__ == "__main__":
    main()
