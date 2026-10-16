# Reject-and-split vs ByteTrack baseline (MEVA G330 / G299)

**Date:** 2026-10-16  
**Clip:** MEVA `2018-03-09.10-15-00.10-20-00.school`, cameras G330 and G299  
**Settings (both runs):** stride 10, crop-every 3, detector score threshold 0.5  

| Run | Tracker | Output |
|---|---|---|
| **Before** | `ByteTrackPersonTracker` only | `perception_runs/meva_baseline_rerun/{G330,G299}/` |
| **After** | `RejectAndSplitPersonTracker` (HSV gate over ByteTrack) | `perception_runs/meva_reject_and_split/{G330,G299}/` |

Local track IDs are **not** stable across runs, so this document does **not** claim a global tracker accuracy rate. It compares aggregate continuity metrics and explains what the gate does for the regression failure modes in `HANDOFF_tracking_owner.md`.

---

## What reject-and-split does

ByteTrack associates detections using motion and IoU only. When two people cross, the Kalman box of one track can land on the other person and the same local ID continues on the wrong body (crossing switches and chain tracks).

**Reject-and-split** keeps ByteTrack as the associator and adds a cheap **local** appearance check:

1. For each emitted track, keep an EMA-updated **HSV histogram** of chromatic pixels in the detector crop (not OSNet, not a P-code, not ReID).
2. When ByteTrack proposes continuing track *T* with a new crop, compare that crop’s histogram to *T*’s signature (`cv2.HISTCMP_CORREL`).
3. If correlation **&lt; 0.45** (`REJECT_AND_SPLIT_SETTINGS["min_correlation"]`), **reject** the continuation and emit a **new** local ID (split). The ByteTrack id is remapped to that new local id for later frames.
4. Fragments are **never** auto-merged. Greyscale / low-chroma crops skip the gate (no split) so blank/white synthetic boxes are not over-split.

**How this is meant to help the MEVA regression:**

| Failure mode | Expected effect of the gate |
|---|---|
| Crossing ID switch (e.g. beige → blue hoodie) | Appearance disagrees → split into two local IDs instead of one chained ID |
| Chain tracks (one ID through 3–6 people) | Each clothing change that drops below the correlation threshold starts a new ID |
| Wrong first association | First crop sets the signature; a different person on the next match tends to split |
| Stable controls (same person, long span) | Similar clothing histograms stay above threshold → ID should remain |
| Long-gap fragmentation (e.g. T360 / T412) | **Not fixed** — merge is out of scope for this change |

Trade-off: fewer impure (multi-person) long tracks, at the cost of **more** local tracks and more short fragments when lighting, occlusion, or loose boxes change the histogram.

---

## Run totals

Observation counts match (gate only remaps IDs; it does not drop detections).

| Camera | Run | Tracks | Observations | Median obs/track | Single-obs tracks | Tracks with span ≥ 10 s | Max span |
|---|---|---:|---:|---:|---:|---:|---:|
| G330 | Baseline | 893 | 18 534 | 7 | 157 | 196 | 129.7 s |
| G330 | Reject-and-split | 1644 | 18 534 | 3 | 476 | 177 | 129.7 s |
| G299 | Baseline | 828 | 17 733 | 7 | 138 | 177 | 132.7 s |
| G299 | Reject-and-split | 1609 | 17 733 | 3 | 483 | 149 | 103.7 s |

**Delta:** G330 **+751** tracks, G299 **+781** tracks; observations unchanged.

### Track span distribution

| Camera | Run | &lt;1 s | 1–3 s | 3–10 s | 10–30 s | 30–60 s | ≥60 s |
|---|---|---:|---:|---:|---:|---:|---:|
| G330 | Baseline | 275 | 191 | 231 | 142 | 44 | 10 |
| G330 | Reject-and-split | 823 | 333 | 311 | 156 | 16 | 5 |
| G299 | Baseline | 247 | 178 | 226 | 132 | 33 | 12 |
| G299 | Reject-and-split | 817 | 339 | 304 | 116 | 24 | 9 |

---

## Appearance purity (main improvement signal)

For every track with at least two saved crops, consecutive crop HSV correlation was scored with the **same** threshold used by the gate (0.45). A track is **impure** if any consecutive pair falls below that threshold (proxy for an ID carrying more than one appearance / person).

| Camera | Run | Tracks scored | Mean of per-track mean corr | Impure tracks | Impure rate | Break events | Impure with span ≥ 10 s |
|---|---|---:|---:|---:|---:|---:|---:|
| G330 | Baseline | 736 | 0.633 | 371 | **50.4%** | 842 | 153 |
| G330 | Reject-and-split | 1166 | 0.688 | 279 | **23.9%** | 402 | 92 |
| G299 | Baseline | 690 | 0.609 | 370 | **53.6%** | 905 | 142 |
| G299 | Reject-and-split | 1126 | 0.673 | 285 | **25.3%** | 429 | 72 |

### Verdict

**Improved on purity / switch chaining:**

- Impure-track rate roughly **halved** on both cameras (~50% → ~24%).
- Total within-track appearance “break” events roughly **halved**.
- Long impure tracks (span ≥ 10 s) dropped (G330 153→92, G299 142→72).
- Mean within-track crop correlation rose (~0.63→0.69 G330, ~0.61→0.67 G299).

**Regressed on continuity length (expected trade-off):**

- ~1.8× more local tracks; many more &lt;1 s / single-observation tracks.
- Fewer very long tracks (≥60 s).

That pattern matches the design: prefer **splitting a wrong continuation** over keeping a multi-person chain. Downstream ReID (Yahya) can still link fragments; a polluted local ID is harder to undo.

Stable-control windows still contain many short IDs in the crowd (both runs); the gate does not invent merges. Manual crop-strip review remains the ground truth for named handoff cases.

---

## Code / config changed

- `perception_minimal/tracker.py`: `RejectAndSplitPersonTracker`, `hsv_histogram`, `REJECT_AND_SPLIT_SETTINGS`
- `perception_minimal/run.py`: default `tracker_factory=RejectAndSplitPersonTracker`; passes `frame_bgr` into `update`
- `BYTETRACK_SETTINGS` **unchanged**
- Unit tests: `tests/perception_minimal/test_reject_and_split.py`

Reproduce the comparison runs:

```bash
# before (already at perception_runs/meva_baseline_rerun)
# after:
OUT_ROOT=perception_runs/meva_reject_and_split bash scripts/run_meva_baseline.sh
```

---

## What this does *not* claim

- Not a MOTA / IDF1 score for MEVA.
- Not a fix for long-gap same-person fragmentation (no merge).
- Not global identity / P-codes.
- Named catalog track IDs from Yahya’s jsonl do not map 1:1 onto either local re-run; use crop purity and strips, not TID equality, for before/after.
