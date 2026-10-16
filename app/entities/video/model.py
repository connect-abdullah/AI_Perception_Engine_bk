"""Video model.

Each row represents one recorded (or pre-recorded) video file.
video_start_time is the wall-clock time when the recording began; it anchors
every frame timestamp so the system can reconstruct absolute event times from
frame numbers.

Nullable technical fields (fps, width, height, duration_seconds) are filled
in by the ingestion pipeline after the file is inspected; they may be absent
for freshly uploaded files.

status tracks the pipeline lifecycle: uploaded → processing → completed|failed.
Only those four values are valid; other strings are rejected by the CHECK.
"""

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Optional

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.entities.camera.model import Camera
    from app.entities.track.model import Track

_VALID_STATUSES = ("uploaded", "processing", "completed", "failed")


class Video(Base):
    __tablename__ = "videos"

    __table_args__ = (
        # Closed enum enforced at DB level; new pipeline states need a migration.
        sa.CheckConstraint(
            "status IN ('uploaded', 'processing', 'completed', 'failed')",
            name="ck_videos_status",
        ),
        # Technical metadata must be positive when present.
        sa.CheckConstraint("fps IS NULL OR fps > 0", name="ck_videos_fps"),
        # Real CV output vs synthetic simulator data – must stay distinguishable.
        sa.CheckConstraint("source IN ('cv', 'simulated')", name="ck_videos_source"),
        sa.CheckConstraint("width IS NULL OR width > 0", name="ck_videos_width"),
        sa.CheckConstraint("height IS NULL OR height > 0", name="ck_videos_height"),
        sa.CheckConstraint(
            "duration_seconds IS NULL OR duration_seconds >= 0",
            name="ck_videos_duration",
        ),
        # Index on camera_id for "all videos from camera X" queries.
        sa.Index("ix_videos_camera_id", "camera_id"),
        # Index on video_start_time for time-range queries.
        sa.Index("ix_videos_start_time", "video_start_time"),
        # Named unique + index so the ingestion layer can look up by handoff id.
        sa.UniqueConstraint("source_video_id", name="uq_videos_source_video_id"),
        sa.Index("ix_videos_source_video_id", "source_video_id"),
    )

    # ── Columns ───────────────────────────────────────────────────────────────
    id: Mapped[uuid.UUID] = mapped_column(
        sa.Uuid(as_uuid=True),
        primary_key=True,
        server_default=sa.text("gen_random_uuid()"),
        default=uuid.uuid4,
    )
    camera_id: Mapped[uuid.UUID] = mapped_column(
        sa.Uuid(as_uuid=True),
        sa.ForeignKey("cameras.id", ondelete="RESTRICT"),
        nullable=False,
    )
    file_path: Mapped[str] = mapped_column(sa.Text, nullable=False)
    video_start_time: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), nullable=False
    )
    fps: Mapped[Optional[float]] = mapped_column(sa.Float, nullable=True)
    width: Mapped[Optional[int]] = mapped_column(sa.Integer, nullable=True)
    height: Mapped[Optional[int]] = mapped_column(sa.Integer, nullable=True)
    duration_seconds: Mapped[Optional[float]] = mapped_column(sa.Float, nullable=True)
    status: Mapped[str] = mapped_column(
        sa.String(20), nullable=False, server_default="uploaded"
    )
    # Stable string identifier written by the perception producer (e.g.
    # "sha256:<hex>").  Matches the source_video_id field in manifest.json and
    # observations.jsonl; nullable so camera-registration rows stay valid before
    # a run is ingested.  Named unique constraint for lookups and migration
    # introspection.
    source_video_id: Mapped[Optional[str]] = mapped_column(sa.Text, nullable=True)
    source: Mapped[str] = mapped_column(
        sa.String(20), nullable=False, server_default="cv", default="cv"
    )
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), server_default=sa.func.now()
    )

    # ── Relationships ─────────────────────────────────────────────────────────
    camera: Mapped["Camera"] = relationship("Camera", back_populates="videos")
    # ON DELETE CASCADE in FK → tracks: deleting a video removes its tracking
    # data completely.
    tracks: Mapped[list["Track"]] = relationship(
        "Track", back_populates="video", passive_deletes=True
    )
