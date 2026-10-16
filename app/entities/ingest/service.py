"""Ingestion and history service.

All database mutations go through this module so the route layer stays thin.

Idempotency contract:
  - Camera is looked up by camera_code; created once.
  - Video is looked up by source_video_id; created or updated on retry.
  - Track is looked up by (video_id, local_track_id); created once.
  - Observation is upserted by (track_id, frame_number): a retry with the same
    frame updates crop_path and activity fields but never creates a duplicate.
  - StateInterval and Event are always inserted (discrete facts).
  - Person is looked up by person_code; created once.

No identity ranking, no pose rules, no P-code generation.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

# Import Base first so all entity models are registered on its metadata before
# any individual model is accessed below.  Without this, the entity modules
# trigger a circular import through app.db.base.
from app.db.base import Base  # noqa: F401 – side-effect: registers all models

from app.entities.camera.model import Camera
from app.entities.event.model import Event
from app.entities.observation.model import Observation
from app.entities.person.model import Person
from app.entities.state_interval.model import StateInterval
from app.entities.track.model import Track
from app.entities.video.model import Video


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _now_utc() -> datetime:
    return datetime.now(tz=timezone.utc)


def _get_or_create_camera(db: Session, camera_code: str) -> Camera:
    camera = db.scalars(
        select(Camera).where(Camera.camera_code == camera_code)
    ).first()
    if camera is None:
        camera = Camera(camera_code=camera_code)
        db.add(camera)
        db.flush()  # assign id without committing
    return camera


def _get_or_create_video(
    db: Session,
    *,
    source_video_id: str,
    camera_id: uuid.UUID,
    fps: float,
    video_start_time: Optional[datetime],
    width: Optional[int],
    height: Optional[int],
) -> Video:
    video = db.scalars(
        select(Video).where(Video.source_video_id == source_video_id)
    ).first()
    if video is None:
        video = Video(
            camera_id=camera_id,
            source_video_id=source_video_id,
            file_path="",               # populated by a separate pipeline step
            video_start_time=video_start_time or _now_utc(),
            fps=fps,
            width=width,
            height=height,
            status="processing",
            source="cv",
        )
        db.add(video)
        db.flush()
    else:
        # Update mutable metadata on retry without changing the identity.
        if fps is not None:
            video.fps = fps
        if width is not None:
            video.width = width
        if height is not None:
            video.height = height
        if video_start_time is not None:
            video.video_start_time = video_start_time
        db.flush()
    return video


def _get_or_create_track(
    db: Session,
    *,
    video_id: uuid.UUID,
    local_track_id: str,
    first_timestamp: datetime,
) -> Track:
    track = db.scalars(
        select(Track).where(
            Track.video_id == video_id,
            Track.local_track_id == local_track_id,
        )
    ).first()
    if track is None:
        track = Track(
            video_id=video_id,
            local_track_id=local_track_id,
            start_time=first_timestamp,
            end_time=None,
            identity_meta={},
        )
        db.add(track)
        db.flush()
    return track


def _upsert_observation(
    db: Session,
    *,
    track_id: uuid.UUID,
    frame_number: int,
    timestamp: datetime,
    bbox_x: int,
    bbox_y: int,
    bbox_width: int,
    bbox_height: int,
    detection_confidence: float,
    crop_path: Optional[str],
) -> None:
    """INSERT … ON CONFLICT (track_id, frame_number) DO UPDATE."""
    stmt = (
        pg_insert(Observation)
        .values(
            track_id=track_id,
            frame_number=frame_number,
            timestamp=timestamp,
            bbox_x=bbox_x,
            bbox_y=bbox_y,
            bbox_width=bbox_width,
            bbox_height=bbox_height,
            detection_confidence=detection_confidence,
            crop_path=crop_path,
        )
        .on_conflict_do_update(
            constraint="uq_observations_track_frame",
            set_={
                "detection_confidence": detection_confidence,
                "crop_path": crop_path,
                "timestamp": timestamp,
                "bbox_x": bbox_x,
                "bbox_y": bbox_y,
                "bbox_width": bbox_width,
                "bbox_height": bbox_height,
            },
        )
    )
    db.execute(stmt)


# ─────────────────────────────────────────────────────────────────────────────
# IngestionService
# ─────────────────────────────────────────────────────────────────────────────

class IngestionService:
    def __init__(self, db: Session) -> None:
        self._db = db

    # ── Perception run ────────────────────────────────────────────────────────

    def ingest_run(
        self,
        *,
        source_video_id: str,
        camera_code: str,
        fps: float,
        video_start_time: Optional[datetime],
        width: Optional[int],
        height: Optional[int],
        observations: list[dict],
    ) -> dict:
        """Upsert camera, video, tracks, and observations in one transaction.

        Args:
            observations: list of dicts matching ObservationIn fields.

        Returns a summary dict used to build RunIngestResponse.
        """
        db = self._db

        # 1. Camera
        camera = _get_or_create_camera(db, camera_code)

        # 2. Video
        video = _get_or_create_video(
            db,
            source_video_id=source_video_id,
            camera_id=camera.id,
            fps=fps,
            video_start_time=video_start_time,
            width=width,
            height=height,
        )

        # 3. Tracks + observations
        # Group by local_track_id to create tracks before observations.
        track_map: dict[str, Track] = {}
        obs_upserted = 0

        for obs in observations:
            ltid = obs["local_track_id"]
            ts_sec = obs["timestamp_seconds"]
            # Compute an absolute timestamp if video_start_time is known.
            if video.video_start_time and video.fps:
                from datetime import timedelta
                obs_ts = video.video_start_time + timedelta(seconds=ts_sec)
            else:
                # Fallback: store as epoch offset from a known reference.
                obs_ts = datetime(
                    2000, 1, 1, tzinfo=timezone.utc
                ).replace(tzinfo=timezone.utc)

            if ltid not in track_map:
                track = _get_or_create_track(
                    db,
                    video_id=video.id,
                    local_track_id=ltid,
                    first_timestamp=obs_ts,
                )
                track_map[ltid] = track

            track = track_map[ltid]
            x, y, w, h = obs["bbox"]
            _upsert_observation(
                db,
                track_id=track.id,
                frame_number=obs["frame_number"],
                timestamp=obs_ts,
                bbox_x=x,
                bbox_y=y,
                bbox_width=w,
                bbox_height=h,
                detection_confidence=obs["detection_confidence"],
                crop_path=obs.get("crop_path"),
            )
            obs_upserted += 1

        # Update video end status once all observations are in.
        if obs_upserted > 0:
            video.status = "completed"

        db.flush()

        return {
            "source_video_id": source_video_id,
            "camera_code": camera_code,
            "video_id": str(video.id),
            "tracks_upserted": len(track_map),
            "observations_upserted": obs_upserted,
        }

    # ── Identity linking ──────────────────────────────────────────────────────

    def link_identity(
        self,
        *,
        source_video_id: str,
        local_track_id: str,
        person_code: Optional[str],
        identity_status: Optional[str],
        identity_confidence: Optional[float],
        identity_meta: Optional[dict],
    ) -> dict:
        """Link (or unlink) a track to a global person.

        Resolves the track by source_video_id + local_track_id.
        A null person_code leaves person_id unresolved.
        """
        db = self._db

        video = db.scalars(
            select(Video).where(Video.source_video_id == source_video_id)
        ).first()
        if video is None:
            raise ValueError(f"No video with source_video_id={source_video_id!r}")

        track = db.scalars(
            select(Track).where(
                Track.video_id == video.id,
                Track.local_track_id == local_track_id,
            )
        ).first()
        if track is None:
            raise ValueError(
                f"No track {local_track_id!r} in video {source_video_id!r}"
            )

        # Resolve or clear person.
        person_id = None
        resolved_code = None
        if person_code is not None:
            person = db.scalars(
                select(Person).where(Person.person_code == person_code)
            ).first()
            if person is None:
                person = Person(person_code=person_code)
                db.add(person)
                db.flush()
            person_id = person.id
            resolved_code = person.person_code

        track.person_id = person_id
        if identity_status is not None:
            track.identity_status = identity_status
        if identity_confidence is not None:
            track.identity_confidence = identity_confidence
        if identity_meta is not None:
            track.identity_meta = identity_meta

        db.flush()

        return {
            "track_id": str(track.id),
            "local_track_id": local_track_id,
            "person_code": resolved_code,
            "identity_status": track.identity_status,
        }

    # ── Activity / state-interval ─────────────────────────────────────────────

    def ingest_activity(
        self,
        *,
        source_video_id: str,
        local_track_id: str,
        activity: str,
        start_time: datetime,
        end_time: Optional[datetime],
        confidence: Optional[float],
    ) -> dict:
        """Insert one state interval for a track."""
        db = self._db

        track = self._resolve_track(source_video_id, local_track_id)

        interval = StateInterval(
            track_id=track.id,
            activity=activity,
            start_time=start_time,
            end_time=end_time,
            confidence=confidence,
        )
        db.add(interval)
        db.flush()

        return {
            "state_interval_id": str(interval.id),
            "local_track_id": local_track_id,
            "activity": activity,
            "start_time": start_time,
            "end_time": end_time,
        }

    # ── Event ─────────────────────────────────────────────────────────────────

    def ingest_event(
        self,
        *,
        source_video_id: Optional[str],
        local_track_id: Optional[str],
        event_type: str,
        timestamp: datetime,
        confidence: Optional[float],
        event_meta: dict,
    ) -> dict:
        """Insert one discrete event."""
        db = self._db

        track_id = None
        if source_video_id and local_track_id:
            track = self._resolve_track(source_video_id, local_track_id)
            track_id = track.id

        event = Event(
            track_id=track_id,
            event_type=event_type,
            timestamp=timestamp,
            confidence=confidence,
            event_meta=event_meta,
        )
        db.add(event)
        db.flush()

        return {
            "event_id": str(event.id),
            "event_type": event_type,
            "timestamp": timestamp,
            "local_track_id": local_track_id,
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _resolve_track(self, source_video_id: str, local_track_id: str) -> Track:
        db = self._db
        video = db.scalars(
            select(Video).where(Video.source_video_id == source_video_id)
        ).first()
        if video is None:
            raise ValueError(f"No video with source_video_id={source_video_id!r}")
        track = db.scalars(
            select(Track).where(
                Track.video_id == video.id,
                Track.local_track_id == local_track_id,
            )
        ).first()
        if track is None:
            raise ValueError(
                f"No track {local_track_id!r} in video {source_video_id!r}"
            )
        return track


# ─────────────────────────────────────────────────────────────────────────────
# HistoryService
# ─────────────────────────────────────────────────────────────────────────────

class HistoryService:
    def __init__(self, db: Session) -> None:
        self._db = db

    def get_track(self, track_id: str) -> Optional[dict]:
        db = self._db
        try:
            uid = uuid.UUID(track_id)
        except ValueError:
            return None
        track = db.scalars(select(Track).where(Track.id == uid)).first()
        if track is None:
            return None
        return self._track_to_dict(track)

    def get_person(self, person_code: str) -> Optional[dict]:
        db = self._db
        person = db.scalars(
            select(Person).where(Person.person_code == person_code)
        ).first()
        if person is None:
            return None
        tracks = db.scalars(
            select(Track).where(Track.person_id == person.id)
        ).all()
        return {
            "id": str(person.id),
            "person_code": person.person_code,
            "known_name": person.known_name,
            "created_at": person.created_at,
            "tracks": [self._track_to_dict(t) for t in tracks],
        }

    def get_camera_tracks(
        self,
        camera_code: str,
        limit: int = 100,
    ) -> Optional[dict]:
        db = self._db
        camera = db.scalars(
            select(Camera).where(Camera.camera_code == camera_code)
        ).first()
        if camera is None:
            return None
        videos = db.scalars(
            select(Video).where(Video.camera_id == camera.id)
        ).all()
        tracks = []
        for video in videos:
            rows = db.scalars(
                select(Track)
                .where(Track.video_id == video.id)
                .limit(limit)
            ).all()
            for t in rows:
                tracks.append(self._track_to_dict(t, source_video_id=video.source_video_id, camera_code=camera_code))
        return {"camera_code": camera_code, "tracks": tracks}

    def query_time_range(
        self,
        *,
        camera_code: Optional[str] = None,
        source_video_id: Optional[str] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 100,
    ) -> dict:
        db = self._db
        stmt = select(Track)

        if source_video_id is not None:
            video = db.scalars(
                select(Video).where(Video.source_video_id == source_video_id)
            ).first()
            if video is None:
                return {"tracks": []}
            stmt = stmt.where(Track.video_id == video.id)
        elif camera_code is not None:
            camera = db.scalars(
                select(Camera).where(Camera.camera_code == camera_code)
            ).first()
            if camera is None:
                return {"tracks": []}
            video_ids = [
                v.id for v in db.scalars(
                    select(Video).where(Video.camera_id == camera.id)
                ).all()
            ]
            stmt = stmt.where(Track.video_id.in_(video_ids))

        if start is not None:
            stmt = stmt.where(Track.start_time >= start)
        if end is not None:
            stmt = stmt.where(Track.start_time <= end)

        tracks = db.scalars(stmt.limit(limit)).all()
        return {"tracks": [self._track_to_dict(t) for t in tracks]}

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _track_to_dict(
        self,
        track: Track,
        *,
        source_video_id: Optional[str] = None,
        camera_code: Optional[str] = None,
    ) -> dict[str, Any]:
        if source_video_id is None or camera_code is None:
            video = self._db.scalars(
                select(Video).where(Video.id == track.video_id)
            ).first()
            if video is not None:
                source_video_id = source_video_id or video.source_video_id
                if camera_code is None:
                    cam = self._db.scalars(
                        select(Camera).where(Camera.id == video.camera_id)
                    ).first()
                    camera_code = cam.camera_code if cam else None

        person_code = None
        if track.person_id is not None:
            person = self._db.scalars(
                select(Person).where(Person.id == track.person_id)
            ).first()
            person_code = person.person_code if person else None

        return {
            "id": str(track.id),
            "local_track_id": track.local_track_id,
            "source_video_id": source_video_id,
            "camera_code": camera_code,
            "person_code": person_code,
            "identity_status": track.identity_status,
            "identity_confidence": track.identity_confidence,
            "identity_meta": track.identity_meta,
            "start_time": track.start_time,
            "end_time": track.end_time,
        }
