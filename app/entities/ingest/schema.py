"""Pydantic schemas for the perception ingestion and history APIs.

These are the I/O contracts for the routes under /api/v1/ingest and
/api/v1/history.  They are deliberately decoupled from the SQLAlchemy models
so the route layer stays thin.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator


# ─────────────────────────────────────────────────────────────────────────────
# Observation (one detection snapshot, matching the handoff contract)
# ─────────────────────────────────────────────────────────────────────────────

class ObservationIn(BaseModel):
    local_track_id: str
    frame_number: int = Field(ge=0)
    timestamp_seconds: float = Field(ge=0.0)
    bbox: tuple[int, int, int, int]  # x, y, width, height
    detection_confidence: float = Field(ge=0.0, le=1.0)
    crop_path: Optional[str] = None


class ObservationRead(BaseModel):
    id: int
    local_track_id: str
    frame_number: int
    timestamp: datetime
    bbox: tuple[int, int, int, int]
    detection_confidence: float
    crop_path: Optional[str]
    activity: Optional[str]
    activity_confidence: Optional[float]
    world_x: Optional[float]
    world_y: Optional[float]
    world_z: Optional[float]

    model_config = {"from_attributes": True}


# ─────────────────────────────────────────────────────────────────────────────
# Perception run ingestion
# ─────────────────────────────────────────────────────────────────────────────

class RunIngestRequest(BaseModel):
    """Body for POST /api/v1/ingest/run.

    A caller may submit a subset of observations (e.g. one camera's worth).
    The endpoint is idempotent: re-submitting the same run updates metadata
    and upserts observations without creating duplicates.
    """

    source_video_id: str
    camera_code: str
    fps: float = Field(gt=0)
    video_start_time: Optional[datetime] = None
    width: Optional[int] = Field(default=None, gt=0)
    height: Optional[int] = Field(default=None, gt=0)
    observations: list[ObservationIn] = Field(default_factory=list)


class RunIngestResponse(BaseModel):
    source_video_id: str
    camera_code: str
    video_id: str          # internal UUID (for subsequent API calls)
    tracks_upserted: int
    observations_upserted: int


# ─────────────────────────────────────────────────────────────────────────────
# Track read
# ─────────────────────────────────────────────────────────────────────────────

class TrackRead(BaseModel):
    id: str
    local_track_id: str
    source_video_id: Optional[str]
    camera_code: Optional[str]
    person_code: Optional[str]      # null when unresolved
    identity_status: Optional[str]
    identity_confidence: Optional[float]
    identity_meta: dict[str, Any]
    start_time: datetime
    end_time: Optional[datetime]

    model_config = {"from_attributes": True}


# ─────────────────────────────────────────────────────────────────────────────
# Identity linking
# ─────────────────────────────────────────────────────────────────────────────

class IdentityLinkRequest(BaseModel):
    """Body for POST /api/v1/ingest/identity.

    Links a local track to a global person.  All fields except the track
    locator and person_code are optional; omitting them leaves the existing
    values unchanged.  To mark a track as unresolved pass person_code=null
    explicitly.
    """

    source_video_id: str
    local_track_id: str
    person_code: Optional[str] = None          # null = unresolved
    identity_status: Optional[str] = None
    identity_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    identity_meta: Optional[dict[str, Any]] = None


class IdentityLinkResponse(BaseModel):
    track_id: str
    local_track_id: str
    person_code: Optional[str]
    identity_status: Optional[str]


# ─────────────────────────────────────────────────────────────────────────────
# Activity / state-interval ingestion
# ─────────────────────────────────────────────────────────────────────────────

class ActivityIngestRequest(BaseModel):
    """Body for POST /api/v1/ingest/activity.

    Persists one state interval produced by Abubakar's activity layer.
    Resolves the track by source_video_id + local_track_id.
    """

    source_video_id: str
    local_track_id: str
    activity: str
    start_time: datetime
    end_time: Optional[datetime] = None
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def end_after_start(self) -> "ActivityIngestRequest":
        if self.end_time is not None and self.end_time < self.start_time:
            raise ValueError("end_time must be >= start_time")
        return self


class ActivityIngestResponse(BaseModel):
    state_interval_id: str
    local_track_id: str
    activity: str
    start_time: datetime
    end_time: Optional[datetime]


# ─────────────────────────────────────────────────────────────────────────────
# Event ingestion
# ─────────────────────────────────────────────────────────────────────────────

class EventIngestRequest(BaseModel):
    """Body for POST /api/v1/ingest/event.

    Persists one local event.  Resolves the track by
    source_video_id + local_track_id (both optional: an event can exist
    without a known track).
    """

    source_video_id: Optional[str] = None
    local_track_id: Optional[str] = None
    event_type: str
    timestamp: datetime
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    event_meta: dict[str, Any] = Field(default_factory=dict)


class EventIngestResponse(BaseModel):
    event_id: str
    event_type: str
    timestamp: datetime
    local_track_id: Optional[str]


# ─────────────────────────────────────────────────────────────────────────────
# History reads
# ─────────────────────────────────────────────────────────────────────────────

class PersonRead(BaseModel):
    id: str
    person_code: str
    known_name: Optional[str]
    created_at: datetime
    tracks: list[TrackRead] = Field(default_factory=list)

    model_config = {"from_attributes": True}


class CameraTracksResponse(BaseModel):
    camera_code: str
    tracks: list[TrackRead]


class TimeRangeQuery(BaseModel):
    """Query parameters for /api/v1/history/time-range."""

    camera_code: Optional[str] = None
    source_video_id: Optional[str] = None
    start: Optional[datetime] = None
    end: Optional[datetime] = None
    limit: int = Field(default=100, ge=1, le=1000)


class TimeRangeResponse(BaseModel):
    tracks: list[TrackRead]
