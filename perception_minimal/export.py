"""Write one processed video in the format of docs/reid_handoff_contract.md.

    <out>/manifest.json
    <out>/observations.jsonl
    <out>/crops/<local_track_id>/<frame_number:08d>.png

Crops are the exact bbox slice of the source frame, saved as lossless PNG. Per track, a
crop is saved for observations 0, K, 2K, ... (K = crop_every) and for the track's last
observation. The last observation is only known when the run ends, so for each track the
crop of its latest unsaved observation is kept in memory (just that crop, never a whole
frame) and written by `finish()`.

Completion: while running, the directory holds `_INCOMPLETE`, `observations.partial.jsonl`
and crops. `finish()` writes `observations.jsonl`, then `manifest.json` (atomically), then
removes the partial file and finally `_INCOMPLETE`. A run directory is complete only if
`manifest.json` exists and `_INCOMPLETE` does not. After a failure the partial files and
crops are left in place, marked by `_INCOMPLETE`, and there is no manifest.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from perception_minimal.tracker import TrackedDetection

CONTRACT_VERSION = "1"
MANIFEST = "manifest.json"
OBSERVATIONS = "observations.jsonl"
PARTIAL_OBSERVATIONS = "observations.partial.jsonl"
INCOMPLETE_MARKER = "_INCOMPLETE"
CROPS = "crops"
_RUN_FILES = {MANIFEST, OBSERVATIONS, PARTIAL_OBSERVATIONS, INCOMPLETE_MARKER, MANIFEST + ".tmp", OBSERVATIONS + ".tmp"}

CAMERA_CODE_MAX_LEN = 50  # backend cameras.camera_code is VARCHAR(50)
_TRACK_ID = re.compile(r"[A-Za-z0-9_-]+")


# --------------------------------------------------------------------------- argument validation


def _check_text(value, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string, got {value!r}")
    if value != value.strip() or any(ord(c) < 32 or ord(c) == 127 for c in value):
        raise ValueError(f"{field} must not have surrounding whitespace or control characters, got {value!r}")
    return value


def check_camera_code(value) -> str:
    _check_text(value, "camera_code")
    if len(value) > CAMERA_CODE_MAX_LEN:
        raise ValueError(f"camera_code must be at most {CAMERA_CODE_MAX_LEN} characters, got {len(value)}")
    return value


def check_source_video_id(value) -> str:
    return _check_text(value, "source_video_id")


def check_positive_int(value, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be an integer >= 1, got {value!r}")
    return value


def parse_video_start_time(value: str | None) -> str | None:
    """None stays None. Otherwise an ISO-8601 datetime with a timezone offset, returned normalised."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"video_start_time must be an ISO-8601 string, got {value!r}")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        raise ValueError(f"video_start_time is not an ISO-8601 datetime: {value!r}") from None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"video_start_time must include a timezone offset, got {value!r}")
    return parsed.isoformat()


def sha256_video_id(path: str | os.PathLike, chunk_size: int = 1 << 20) -> str:
    """Deterministic default source_video_id: "sha256:" + SHA-256 of the file's bytes."""
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _track_sort_key(local_track_id: str):
    return (0, int(local_track_id), "") if local_track_id.isdigit() else (1, 0, local_track_id)


# --------------------------------------------------------------------------- output directory


def prepare_output_dir(out_dir: str | os.PathLike, overwrite: bool) -> Path:
    """Create `out_dir`, or refuse if it is a non-empty directory.

    With `overwrite=True`, a directory holding only files from a previous run (manifest,
    observations, marker, crops/<id>/*.png) is cleared; anything else there is a conflict.
    """
    out = Path(out_dir)
    if out.exists() and not out.is_dir():
        raise FileExistsError(f"Output path exists and is not a directory: {out}")
    if out.is_dir() and any(out.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is not empty: {out} (use --overwrite / overwrite=True to replace a previous run)")
        unknown = [p.name for p in out.iterdir() if p.name not in _RUN_FILES and p.name != CROPS]
        crops = out / CROPS
        if crops.exists():
            if not crops.is_dir():
                unknown.append(CROPS)
            else:
                for p in crops.rglob("*"):
                    rel = p.relative_to(crops)
                    is_track_dir = p.is_dir() and len(rel.parts) == 1
                    is_crop = p.is_file() and len(rel.parts) == 2 and p.suffix == ".png"
                    if not (is_track_dir or is_crop):
                        unknown.append(str(Path(CROPS) / rel))
        if unknown:
            raise FileExistsError(f"Refusing to overwrite {out}: it contains files that are not from a run: {sorted(unknown)}")
        for name in _RUN_FILES:
            (out / name).unlink(missing_ok=True)
        if crops.is_dir():
            shutil.rmtree(crops)
    out.mkdir(parents=True, exist_ok=True)
    return out


# --------------------------------------------------------------------------- writer


@dataclass
class _TrackSummary:
    first_frame_number: int
    first_timestamp_seconds: float
    last_frame_number: int
    last_timestamp_seconds: float
    num_observations: int = 0
    num_crops: int = 0
    pending_crop: np.ndarray | None = None  # crop of the latest observation if not yet saved

    def as_dict(self, local_track_id: str) -> dict:
        return {
            "local_track_id": local_track_id,
            "first_frame_number": self.first_frame_number,
            "last_frame_number": self.last_frame_number,
            "first_timestamp_seconds": self.first_timestamp_seconds,
            "last_timestamp_seconds": self.last_timestamp_seconds,
            "num_observations": self.num_observations,
            "num_crops": self.num_crops,
        }


class RunWriter:
    """Streams observations and crops for one video into `out_dir`; `finish()` completes it.

    Use as a context manager: leaving the block without `finish()` aborts the run.
    """

    def __init__(self, out_dir: str | os.PathLike, *, source_video_id: str, crop_every: int, overwrite: bool = False):
        self.source_video_id = check_source_video_id(source_video_id)
        self.crop_every = check_positive_int(crop_every, "crop_every")
        self.out = prepare_output_dir(out_dir, overwrite)
        (self.out / INCOMPLETE_MARKER).write_text("This run did not complete; do not use its contents.\n")
        self._partial = open(self.out / PARTIAL_OBSERVATIONS, "w", encoding="utf-8", newline="\n")
        self._tracks: dict[str, _TrackSummary] = {}
        self._last_frame_number: int | None = None
        self._closed = False

    # -- per frame

    def add_frame(self, frame_number: int, timestamp_seconds: float, frame_bgr: np.ndarray,
                  tracked: list[TrackedDetection]) -> None:
        if self._closed:
            raise RuntimeError("RunWriter is closed")
        if isinstance(frame_number, bool) or not isinstance(frame_number, int) or frame_number < 0:
            raise ValueError(f"frame_number must be an integer >= 0, got {frame_number!r}")
        if self._last_frame_number is not None and frame_number <= self._last_frame_number:
            raise ValueError(f"Frames must arrive in increasing order: {frame_number} after {self._last_frame_number}")
        if not isinstance(timestamp_seconds, float) or not math.isfinite(timestamp_seconds) or timestamp_seconds < 0:
            raise ValueError(f"timestamp_seconds must be a finite float >= 0, got {timestamp_seconds!r}")
        height, width = frame_bgr.shape[:2]
        self._check_tracked(tracked, width, height)
        self._last_frame_number = frame_number

        for t in sorted(tracked, key=lambda t: _track_sort_key(t.local_track_id)):
            x, y, w, h = t.bbox
            crop = frame_bgr[y:y + h, x:x + w]
            summary = self._tracks.get(t.local_track_id)
            if summary is None:
                summary = self._tracks[t.local_track_id] = _TrackSummary(
                    frame_number, timestamp_seconds, frame_number, timestamp_seconds)
            index = summary.num_observations
            summary.num_observations += 1
            summary.last_frame_number, summary.last_timestamp_seconds = frame_number, timestamp_seconds

            crop_path = None
            if index % self.crop_every == 0:
                crop_path = self._write_crop(t.local_track_id, frame_number, crop)
                summary.num_crops += 1
                summary.pending_crop = None
            else:
                summary.pending_crop = crop.copy()  # only this crop is retained, replacing the previous one

            self._partial.write(json.dumps({
                "source_video_id": self.source_video_id,
                "local_track_id": t.local_track_id,
                "frame_number": frame_number,
                "timestamp_seconds": timestamp_seconds,
                "bbox": [x, y, w, h],
                "detection_confidence": t.detection_score,
                "crop_path": crop_path,
            }) + "\n")

    @staticmethod
    def _check_tracked(tracked, width: int, height: int) -> None:
        if not isinstance(tracked, list):
            raise TypeError(f"Tracker result must be a list, got {type(tracked).__name__}")
        seen = set()
        for t in tracked:
            if not isinstance(t, TrackedDetection):
                raise TypeError(f"Tracker result items must be TrackedDetection, got {type(t).__name__}")
            if not isinstance(t.local_track_id, str) or not _TRACK_ID.fullmatch(t.local_track_id):
                raise ValueError(f"local_track_id must match [A-Za-z0-9_-]+, got {t.local_track_id!r}")
            if t.local_track_id in seen:
                raise ValueError(f"local_track_id {t.local_track_id!r} appears twice in one frame")
            seen.add(t.local_track_id)
            box = t.bbox
            if not (isinstance(box, tuple) and len(box) == 4 and all(type(v) is int for v in box)):
                raise ValueError(f"bbox must be a tuple of 4 ints, got {box!r}")
            x, y, w, h = box
            if w < 1 or h < 1 or x < 0 or y < 0 or x + w > width or y + h > height:
                raise ValueError(f"bbox {box} is not a positive box inside the {width}x{height} frame")
            s = t.detection_score
            if isinstance(s, bool) or not isinstance(s, (int, float)) or not math.isfinite(s) or not 0 <= s <= 1:
                raise ValueError(f"detection_score must be a finite number in [0, 1], got {s!r}")

    def _write_crop(self, local_track_id: str, frame_number: int, crop: np.ndarray) -> str:
        rel = f"{CROPS}/{local_track_id}/{frame_number:08d}.png"
        ok, encoded = cv2.imencode(".png", crop)
        if not ok or encoded is None:
            raise OSError(f"Could not encode crop {rel}")
        path = self.out / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "xb") as f:  # never replace an existing crop
            f.write(encoded.tobytes())
        return rel

    # -- completion

    def finish(self, manifest_fields: dict) -> dict:
        """Write remaining last-observation crops, observations.jsonl and manifest.json."""
        if self._closed:
            raise RuntimeError("RunWriter is closed")
        self._partial.close()

        final_crops: dict[tuple[str, int], str] = {}
        for track_id, summary in self._tracks.items():
            if summary.pending_crop is not None:
                rel = self._write_crop(track_id, summary.last_frame_number, summary.pending_crop)
                final_crops[(track_id, summary.last_frame_number)] = rel
                summary.num_crops += 1
                summary.pending_crop = None
            if summary.num_crops < 1:
                raise RuntimeError(f"Track {track_id} has no crop")

        tmp = self.out / (OBSERVATIONS + ".tmp")
        with open(self.out / PARTIAL_OBSERVATIONS, encoding="utf-8") as src, \
                open(tmp, "w", encoding="utf-8", newline="\n") as dst:
            for line in src:
                obs = json.loads(line)
                key = (obs["local_track_id"], obs["frame_number"])
                if key in final_crops:
                    obs["crop_path"] = final_crops[key]
                dst.write(json.dumps(obs) + "\n")
        os.replace(tmp, self.out / OBSERVATIONS)

        manifest = {"contract_version": CONTRACT_VERSION, "source_video_id": self.source_video_id}
        clash = set(manifest_fields) & {*manifest, "num_observations", "tracks"}
        if clash:
            raise ValueError(f"manifest_fields may not set {sorted(clash)}")
        manifest.update(manifest_fields)
        manifest["num_observations"] = sum(s.num_observations for s in self._tracks.values())
        manifest["tracks"] = [self._tracks[k].as_dict(k) for k in sorted(self._tracks, key=_track_sort_key)]
        tmp = self.out / (MANIFEST + ".tmp")
        tmp.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, self.out / MANIFEST)

        (self.out / PARTIAL_OBSERVATIONS).unlink()
        (self.out / INCOMPLETE_MARKER).unlink()  # last: from here on the run counts as complete
        self._closed = True
        return manifest

    def __enter__(self) -> RunWriter:
        return self

    def __exit__(self, *exc) -> None:
        if not self._closed:  # leaving without finish(): treat as failed
            self.abort()

    def abort(self) -> None:
        """Stop after a failure. Partial output stays, marked incomplete; no manifest is written."""
        if not self._partial.closed:
            self._partial.close()
        for track in self._tracks.values():
            track.pending_crop = None
        self._closed = True
