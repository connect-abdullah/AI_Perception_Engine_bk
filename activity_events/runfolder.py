"""Read a complete perception run folder (docs/reid_handoff_contract.md, contract_version "1")."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

SUPPORTED_CONTRACT_VERSIONS = {"1"}


class RunFolderError(ValueError):
    """The directory is not a complete, readable run folder."""


@dataclass(frozen=True)
class Observation:
    local_track_id: str
    frame_number: int
    timestamp_seconds: float
    bbox: tuple[int, int, int, int]  # x, y, width, height
    detection_confidence: float
    crop_path: str | None

    @property
    def foot_point(self) -> tuple[float, float]:
        """Bottom-centre of the box: where the person stands on the floor in the image."""
        x, y, w, h = self.bbox
        return (x + w / 2.0, float(y + h))


@dataclass
class Track:
    local_track_id: str
    observations: list[Observation] = field(default_factory=list)

    @property
    def start_seconds(self) -> float:
        return self.observations[0].timestamp_seconds

    @property
    def end_seconds(self) -> float:
        return self.observations[-1].timestamp_seconds

    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds


@dataclass
class RunFolder:
    path: Path
    manifest: dict
    tracks: dict[str, Track]

    @property
    def source_video_id(self) -> str:
        return self.manifest["source_video_id"]

    @property
    def camera_code(self) -> str:
        return self.manifest["camera_code"]

    @property
    def fps(self) -> float:
        return float(self.manifest["fps"])

    @property
    def frame_size(self) -> tuple[int, int] | None:
        w, h = self.manifest.get("width"), self.manifest.get("height")
        return (int(w), int(h)) if w and h else None

    @property
    def video_start_time(self) -> datetime | None:
        value = self.manifest.get("video_start_time")
        return datetime.fromisoformat(value) if value else None

    def absolute_time(self, seconds: float) -> str | None:
        """ISO-8601 wall-clock time for an offset into this video, or None if the start is unknown.

        Never guessed: without manifest.video_start_time there is no absolute time.
        """
        start = self.video_start_time
        return (start + timedelta(seconds=seconds)).isoformat() if start else None

    def crop_file(self, obs: Observation) -> Path | None:
        return self.path / obs.crop_path if obs.crop_path else None


def load_run(run_dir: str | Path) -> RunFolder:
    """Load a run folder, refusing incomplete or unknown-version runs (contract 'Completeness')."""
    path = Path(run_dir)
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        raise RunFolderError(f"{path}: no manifest.json, so the run is not complete")
    if (path / "_INCOMPLETE").exists():
        raise RunFolderError(f"{path}: _INCOMPLETE marker present, so the run is not complete")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    version = str(manifest.get("contract_version"))
    if version not in SUPPORTED_CONTRACT_VERSIONS:
        raise RunFolderError(f"{path}: unsupported contract_version {version!r}")
    for key in ("source_video_id", "camera_code", "fps"):
        if key not in manifest:
            raise RunFolderError(f"{path}: manifest.json is missing {key!r}")

    tracks: dict[str, Track] = {}
    obs_path = path / "observations.jsonl"
    if not obs_path.is_file():
        raise RunFolderError(f"{path}: no observations.jsonl")
    with obs_path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("source_video_id") != manifest["source_video_id"]:
                raise RunFolderError(f"{obs_path}:{line_no}: source_video_id differs from the manifest")
            obs = Observation(
                local_track_id=str(row["local_track_id"]),
                frame_number=int(row["frame_number"]),
                timestamp_seconds=float(row["timestamp_seconds"]),
                bbox=tuple(int(v) for v in row["bbox"]),
                detection_confidence=float(row["detection_confidence"]),
                crop_path=row.get("crop_path"),
            )
            tracks.setdefault(obs.local_track_id, Track(obs.local_track_id)).observations.append(obs)

    for track in tracks.values():
        track.observations.sort(key=lambda o: o.frame_number)
    return RunFolder(path=path, manifest=manifest, tracks=tracks)
