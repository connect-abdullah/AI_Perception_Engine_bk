"""Offline perception run: video -> person detection -> local tracking -> handoff contract files.

    python -m perception_minimal.run VIDEO --camera-code CAM_A --out perception_runs/example \\
        --score-threshold <value> --stride <N> --crop-every <K> \\
        [--source-video-id ID] [--video-start-time 2026-10-14T09:00:00+05:00] [--overwrite]

Every processing setting is an explicit argument; none has a site-specific default. The
detector score threshold is not calibrated. Without --source-video-id the id is
"sha256:" + the SHA-256 of the video file's bytes, so the same file always gets the same id.
Without --video-start-time the manifest records null; it is never guessed from file dates.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable
from importlib import metadata
from pathlib import Path

from perception_minimal.detector import PersonDetector, check_score_threshold
from perception_minimal.export import (
    RunWriter,
    check_camera_code,
    check_positive_int,
    check_source_video_id,
    parse_video_start_time,
    prepare_output_dir,
    sha256_video_id,
)
from perception_minimal.tracker import PersonTracker, RejectAndSplitPersonTracker
from perception_minimal.video import VideoReader

PRODUCER = "perception_minimal"
_PACKAGES = ("numpy", "opencv-python-headless", "torch", "torchvision", "supervision", "scipy")


def _package_versions() -> dict[str, str | None]:
    versions = {}
    for name in _PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def run_pipeline(
    video_path: str | os.PathLike,
    *,
    out_dir: str | os.PathLike,
    camera_code: str,
    detector: PersonDetector,
    stride: int,
    crop_every: int,
    source_video_id: str | None = None,
    video_start_time: str | None = None,
    overwrite: bool = False,
    tracker_factory: Callable[[float], PersonTracker] = RejectAndSplitPersonTracker,
) -> dict:
    """Process one video into `out_dir`; return the manifest written there.

    The tracker is created with frame rate = video fps / stride (the rate it receives
    updates). On any failure the exception propagates, the writer is aborted, no manifest
    is written and the directory keeps its `_INCOMPLETE` marker.
    """
    camera_code = check_camera_code(camera_code)
    stride = check_positive_int(stride, "stride")
    crop_every = check_positive_int(crop_every, "crop_every")
    video_start_time = parse_video_start_time(video_start_time)
    if source_video_id is not None:
        source_video_id = check_source_video_id(source_video_id)

    with VideoReader(video_path) as video:  # rejects a missing or undecodable video before any output exists
        meta = video.metadata
        if source_video_id is None:
            source_video_id = sha256_video_id(meta.path)
        tracker_frame_rate = meta.fps / stride
        tracker = tracker_factory(tracker_frame_rate)
        with RunWriter(out_dir, source_video_id=source_video_id, crop_every=crop_every, overwrite=overwrite) as writer:
            processed = 0
            last_frame_number = None
            for frame in video.frames(stride):
                detections = detector.detect(frame.frame_bgr)
                tracked = tracker.update(detections, frame.frame_bgr)
                writer.add_frame(frame.frame_number, frame.timestamp_seconds, frame.frame_bgr, tracked)
                processed += 1
                last_frame_number = frame.frame_number

            return writer.finish({
                "camera_code": camera_code,
                "fps": meta.fps,
                "video_start_time": video_start_time,
                "width": meta.width,
                "height": meta.height,
                "frame_count": meta.reported_frame_count,
                "timestamp_method": meta.timestamp_method,
                "processing": {
                    "stride": stride,
                    "crop_every": crop_every,
                    "tracker_frame_rate": tracker_frame_rate,
                    "processed_frames": processed,
                    "last_processed_frame_number": last_frame_number,
                },
                "producer": {
                    "name": PRODUCER,
                    "detector": getattr(detector, "provenance", None) or {"class": type(detector).__name__},
                    "score_threshold": getattr(detector, "score_threshold", None),
                    "tracker": getattr(tracker, "config", None) or {"class": type(tracker).__name__},
                    "packages": _package_versions(),
                },
            })


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m perception_minimal.run", description=__doc__.split("\n\n")[0])
    p.add_argument("video", help="local video file")
    p.add_argument("--camera-code", required=True, help="readable camera code, e.g. CAM-LOBBY-01 (not a database UUID)")
    p.add_argument("--out", required=True, help="output run directory; must be new or empty")
    p.add_argument("--score-threshold", required=True, type=float, help="detector score threshold in [0, 1] (not calibrated)")
    p.add_argument("--stride", required=True, type=int, help="process every N-th source frame (N >= 1)")
    p.add_argument("--crop-every", required=True, type=int, help="save every K-th observation's crop per track (K >= 1)")
    p.add_argument("--source-video-id", help="default: sha256:<SHA-256 of the video file>")
    p.add_argument("--video-start-time", help="ISO-8601 datetime of frame 0 with timezone offset; default null")
    p.add_argument("--overwrite", action="store_true", help="replace a previous run in --out (only run files may be present)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        # Validate everything and check the inputs/output before loading the model.
        check_camera_code(args.camera_code)
        threshold = check_score_threshold(args.score_threshold)
        check_positive_int(args.stride, "stride")
        check_positive_int(args.crop_every, "crop_every")
        parse_video_start_time(args.video_start_time)
        if args.source_video_id is not None:
            check_source_video_id(args.source_video_id)
        with VideoReader(args.video):
            pass
        out = Path(args.out)
        if out.exists() and (not out.is_dir() or any(out.iterdir())) and not args.overwrite:
            prepare_output_dir(out, overwrite=False)  # raises the conflict error

        from perception_minimal.detector import TorchvisionPersonDetector

        manifest = run_pipeline(
            args.video,
            out_dir=out,
            camera_code=args.camera_code,
            detector=TorchvisionPersonDetector(threshold),
            stride=args.stride,
            crop_every=args.crop_every,
            source_video_id=args.source_video_id,
            video_start_time=args.video_start_time,
            overwrite=args.overwrite,
        )
    except (ValueError, TypeError, OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {out}: {len(manifest['tracks'])} tracks, {manifest['num_observations']} observations, "
          f"{manifest['processing']['processed_frames']} processed frames")
    return 0


if __name__ == "__main__":
    sys.exit(main())
