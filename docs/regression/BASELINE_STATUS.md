# MEVA regression baseline status

**Date:** 2026-10-15  
**Branch:** `abdullah/perception`  
**Commit:** `b40d4e6` (persistence layer) + local eval tooling

## Inputs provided

| Artifact | Path |
|---|---|
| G330 video | `eval_data/meva/G330_source_video.avi` (9008 frames, 30 fps, 1920×1080) |
| G299 video | `eval_data/meva/G299_source_video.avi` (9010 frames, 30 fps, 1920×1080) |
| Yahya baseline obs | `eval_data/meva/baseline/{G330,G299}_observations.jsonl` |

## Yahya baseline vs regression catalog

Command:

```bash
python scripts/eval_meva_regression.py \
  --g330 eval_data/meva/baseline/G330_observations.jsonl \
  --g299 eval_data/meva/baseline/G299_observations.jsonl \
  --label baseline
```

Result: **27 / 27 priority tracks present** (crossings, chains, wrong-first, fragmentation, stable controls, short tracks). Frame windows match the handoff.

This confirms the supplied baseline is the correct regression target. It does **not** measure improvement yet — switch/chain verdicts still need crop-strip review.

## Smoke test (live pipeline on real G330)

- Window: frames 0–190, stride 10, score_threshold 0.5
- Result: 20 frames, 278 observations, 23 local tracks, ~3.5 s/frame on CPU
- Output: `perception_runs/meva_smoke_G330/`
- Full-clip estimate: ~53 min per camera on this machine

## Full re-run (complete)

Finished 2026-10-16 00:49 via `bash scripts/run_meva_baseline.sh` (~57 min/camera, CPU).

| Camera | Tracks | Obs | Time |
|---|---:|---:|---:|
| G330 | 893 | 18534 | 56.9m |
| G299 | 828 | 17733 | 56.0m |

Outputs: `perception_runs/meva_baseline_rerun/{G330,G299}/` (manifest + observations.jsonl + crops).

Yahya baseline totals are close (G330 881/18583, G299 822/17792). Named local-track IDs are **not** stable across runs (same ids exist but spans differ), so use Yahya’s jsonl for the catalog TID cases and this re-run + its crops as the before-gate visual baseline.

## Tracker change status

**Applied:** `RejectAndSplitPersonTracker` (HSV histogram gate) wraps ByteTrack behind the
`PersonTracker` seam. Default in `perception_minimal.run`. Settings in
`REJECT_AND_SPLIT_SETTINGS` (`min_correlation=0.45`). No auto-merge of fragments.
`BYTETRACK_SETTINGS` unchanged.

Gated re-run complete: `perception_runs/meva_reject_and_split/{G330,G299}/`  
Before/after write-up: [REJECT_AND_SPLIT_COMPARISON.md](REJECT_AND_SPLIT_COMPARISON.md)
