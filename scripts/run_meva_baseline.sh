#!/usr/bin/env bash
# Sequential MEVA re-run with visible progress.
# Usage:
#   bash scripts/run_meva_baseline.sh
#   TRACKER=bytetrack OUT_ROOT=perception_runs/meva_baseline_rerun bash scripts/run_meva_baseline.sh
#   TRACKER=reject_and_split OUT_ROOT=perception_runs/meva_reject_and_split_checkpoint bash scripts/run_meva_baseline.sh
#
# TRACKER: reject_and_split (default) | bytetrack
set -euo pipefail
cd "$(dirname "$0")/.."

VENV="${VENV:-/private/var/folders/4f/57tj8gl55_7139f56dw52qr40000gn/T/cursor-sandbox-cache/e94cf220e2bf98de6cc1b9d37cab4b10/poetry/virtualenvs/ai-perception-engine-sTO6kafp-py3.14}"
PY="${VENV}/bin/python"
OUT_ROOT="${OUT_ROOT:-perception_runs/meva_reject_and_split}"
TRACKER="${TRACKER:-reject_and_split}"
case "${TRACKER}" in
  reject_and_split|bytetrack) ;;
  *)
    echo "TRACKER must be reject_and_split or bytetrack, got: ${TRACKER}" >&2
    exit 1
    ;;
esac
mkdir -p "$OUT_ROOT"
echo "OUT_ROOT=${OUT_ROOT}"
echo "TRACKER=${TRACKER}"

run_one() {
  local cam="$1" video="$2" sid="$3"
  local out="${OUT_ROOT}/${cam}"
  echo "============================================================"
  echo "START ${cam}  $(date '+%H:%M:%S')"
  echo "  video: ${video}"
  echo "  out:   ${out}"
  echo "  tracker: ${TRACKER}"
  echo "============================================================"
  rm -rf "$out"
  TRACKER_MODE="${TRACKER}" PYTHONUNBUFFERED=1 "$PY" -u - <<PY
import os, sys, time
from pathlib import Path
from perception_minimal.detector import TorchvisionPersonDetector
from perception_minimal.tracker import ByteTrackPersonTracker, RejectAndSplitPersonTracker
from perception_minimal.video import VideoReader
from perception_minimal.export import RunWriter, prepare_output_dir

cam = "${cam}"
video = "${video}"
out = Path("${out}")
sid = "${sid}"
mode = os.environ["TRACKER_MODE"]
STRIDE, CROP_EVERY, THRESH = 10, 3, 0.5

print(f"[{cam}] loading detector...", flush=True)
det = TorchvisionPersonDetector(score_threshold=THRESH)
print(f"[{cam}] detector ready", flush=True)

t0 = time.time()
with VideoReader(video) as vr:
    total = vr.metadata.reported_frame_count or 0
    approx = (total // STRIDE) if total else "?"
    print(f"[{cam}] {vr.metadata.width}x{vr.metadata.height} @ {vr.metadata.fps}fps  "
          f"frames≈{total}  processed≈{approx}", flush=True)
    frame_rate = vr.metadata.fps / STRIDE
    if mode == "bytetrack":
        tracker = ByteTrackPersonTracker(frame_rate)
        gate = None
        min_corr = None
    else:
        tracker = RejectAndSplitPersonTracker(frame_rate)
        gate = tracker.config.get("gate")
        min_corr = tracker.config.get("min_correlation")
    print(f"[{cam}] tracker={tracker.config.get('tracker', type(tracker).__name__)} "
          f"gate={gate} min_corr={min_corr}", flush=True)
    prepare_output_dir(out, overwrite=True)
    with RunWriter(out, source_video_id=sid, crop_every=CROP_EVERY, overwrite=True) as writer:
        n = 0
        for frame in vr.frames(STRIDE):
            detections = det.detect(frame.frame_bgr)
            tracked = tracker.update(detections, frame.frame_bgr)
            writer.add_frame(frame.frame_number, frame.timestamp_seconds, frame.frame_bgr, tracked)
            n += 1
            if n == 1 or n % 10 == 0:
                elapsed = time.time() - t0
                rate = elapsed / n
                eta = (approx - n) * rate if isinstance(approx, int) else float("nan")
                print(
                    f"[{cam}] frame {frame.frame_number:>5}  "
                    f"processed {n:>4}/{approx}  "
                    f"dets={len(detections):2} tracks={len(tracked):2}  "
                    f"{rate:.1f}s/f  ETA {eta/60:.0f}m",
                    flush=True,
                )
        manifest = writer.finish({
            "camera_code": cam,
            "fps": vr.metadata.fps,
            "video_start_time": None,
            "width": vr.metadata.width,
            "height": vr.metadata.height,
            "frame_count": vr.metadata.reported_frame_count,
            "timestamp_method": vr.metadata.timestamp_method,
            "processing": {
                "stride": STRIDE,
                "crop_every": CROP_EVERY,
                "tracker_frame_rate": vr.metadata.fps / STRIDE,
                "processed_frames": n,
                "last_processed_frame_number": frame.frame_number,
                "tracker_mode": mode,
            },
            "producer": {
                "name": "perception_minimal",
                "score_threshold": THRESH,
                "tracker": tracker.config,
            },
        })
elapsed = time.time() - t0
print(
    f"[{cam}] DONE in {elapsed/60:.1f}m  "
    f"tracks={len(manifest['tracks'])}  "
    f"obs={manifest['num_observations']}  "
    f"frames={manifest['processing']['processed_frames']}",
    flush=True,
)
PY
}

run_one G330 \
  eval_data/meva/G330_source_video.avi \
  meva_2018-03-09.10-15-00.10-20-00.school.G330

run_one G299 \
  eval_data/meva/G299_source_video.avi \
  meva_2018-03-09.10-15-00.10-20-00.school.G299

echo "ALL DONE $(date '+%H:%M:%S')"
echo "Outputs: ${OUT_ROOT}/G330  and  ${OUT_ROOT}/G299"
echo "TRACKER=${TRACKER}"
