#!/usr/bin/env python3
"""Compare aggregate track continuity metrics between two perception run roots.

Same method as docs/regression/REJECT_AND_SPLIT_COMPARISON.md:
  - track counts, span histograms, single-obs counts
  - consecutive-crop HSV impurity (threshold = REJECT_AND_SPLIT_SETTINGS min_correlation)

Usage:
  python scripts/compare_meva_purity.py \\
    --before perception_runs/meva_baseline_rerun \\
    --after  perception_runs/meva_reject_and_split_checkpoint \\
    --cameras G330 G299
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from perception_minimal.tracker import REJECT_AND_SPLIT_SETTINGS, histogram_correlation, hsv_histogram


def _load_obs(run_cam: Path) -> list[dict]:
    path = run_cam / "observations.jsonl"
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _track_spans(obs: list[dict]) -> dict[str, dict]:
    tracks: dict[str, dict] = {}
    for o in obs:
        tid = o["local_track_id"]
        t = o["timestamp_seconds"]
        if tid not in tracks:
            tracks[tid] = {
                "first": t,
                "last": t,
                "n": 0,
                "crops": [],
            }
        tr = tracks[tid]
        tr["first"] = min(tr["first"], t)
        tr["last"] = max(tr["last"], t)
        tr["n"] += 1
        if o.get("crop_path"):
            tr["crops"].append((o["frame_number"], o["crop_path"], o["bbox"]))
    return tracks


def _span_bucket(span: float) -> str:
    if span < 1:
        return "<1s"
    if span < 3:
        return "1-3s"
    if span < 10:
        return "3-10s"
    if span < 30:
        return "10-30s"
    if span < 60:
        return "30-60s"
    return ">=60s"


def _score_purity(run_cam: Path, tracks: dict[str, dict], threshold: float) -> dict:
    scored = 0
    impure = 0
    break_events = 0
    impure_long = 0
    mean_corrs: list[float] = []

    for tid, tr in tracks.items():
        crops = sorted(tr["crops"], key=lambda x: x[0])
        if len(crops) < 2:
            continue
        corrs: list[float] = []
        prev_hist = None
        for _fn, rel, bbox in crops:
            crop_path = run_cam / rel
            if not crop_path.is_file():
                continue
            img = cv2.imread(str(crop_path), cv2.IMREAD_COLOR)
            if img is None:
                continue
            # Crop files are already the bbox slice; hist over full crop.
            h, w = img.shape[:2]
            hist = hsv_histogram(img, (0, 0, w, h))
            if hist is None:
                continue
            if prev_hist is not None:
                c = histogram_correlation(prev_hist, hist)
                corrs.append(c)
                if c < threshold:
                    break_events += 1
            prev_hist = hist
        if not corrs:
            continue
        scored += 1
        mean_corrs.append(float(np.mean(corrs)))
        if any(c < threshold for c in corrs):
            impure += 1
            if (tr["last"] - tr["first"]) >= 10.0:
                impure_long += 1

    return {
        "tracks_scored": scored,
        "mean_of_per_track_mean_corr": float(np.mean(mean_corrs)) if mean_corrs else float("nan"),
        "impure_tracks": impure,
        "impure_rate": (impure / scored) if scored else float("nan"),
        "break_events": break_events,
        "impure_span_ge_10s": impure_long,
    }


def summarize(run_cam: Path, threshold: float) -> dict:
    obs = _load_obs(run_cam)
    tracks = _track_spans(obs)
    spans = [tr["last"] - tr["first"] for tr in tracks.values()]
    single = sum(1 for tr in tracks.values() if tr["n"] == 1)
    buckets = defaultdict(int)
    for s in spans:
        buckets[_span_bucket(s)] += 1
    med_obs = float(np.median([tr["n"] for tr in tracks.values()])) if tracks else float("nan")
    purity = _score_purity(run_cam, tracks, threshold)
    return {
        "tracks": len(tracks),
        "observations": len(obs),
        "median_obs_per_track": med_obs,
        "single_obs_tracks": single,
        "tracks_span_ge_10s": sum(1 for s in spans if s >= 10),
        "max_span": max(spans) if spans else 0.0,
        "span_buckets": dict(buckets),
        **purity,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--before", type=Path, required=True)
    p.add_argument("--after", type=Path, required=True)
    p.add_argument("--cameras", nargs="+", default=["G330", "G299"])
    p.add_argument(
        "--threshold",
        type=float,
        default=REJECT_AND_SPLIT_SETTINGS["min_correlation"],
    )
    args = p.parse_args()

    print(f"threshold={args.threshold}")
    print(f"before={args.before}")
    print(f"after={args.after}")
    print()

    for cam in args.cameras:
        before = summarize(args.before / cam, args.threshold)
        after = summarize(args.after / cam, args.threshold)
        print(f"=== {cam} ===")
        for label, s in (("before", before), ("after", after)):
            print(
                f"  {label}: tracks={s['tracks']} obs={s['observations']} "
                f"median_obs/track={s['median_obs_per_track']:.1f} "
                f"single_obs={s['single_obs_tracks']} "
                f"span>=10s={s['tracks_span_ge_10s']} max_span={s['max_span']:.1f}s"
            )
            print(
                f"         spans={s['span_buckets']} "
                f"impure={s['impure_tracks']}/{s['tracks_scored']} "
                f"({100 * s['impure_rate']:.1f}%) "
                f"breaks={s['break_events']} "
                f"impure>=10s={s['impure_span_ge_10s']} "
                f"mean_corr={s['mean_of_per_track_mean_corr']:.3f}"
            )
        print(
            f"  delta tracks: {after['tracks'] - before['tracks']:+d}  "
            f"impure rate: {100 * before['impure_rate']:.1f}% → {100 * after['impure_rate']:.1f}%"
        )
        print()


if __name__ == "__main__":
    main()
