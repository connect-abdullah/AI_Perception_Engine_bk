"""Track model.

A track is the output of a tracking algorithm (e.g. ByteTrack) for one
continuous detection session within a single video.  local_track_id is the
ID assigned by the tracker; it is only unique within its video, so the
composite unique constraint (video_id, local_track_id) prevents collisions
when multiple videos are processed.

person_id is nullable: the ReID stage assigns it after the track is created.
Tracks whose person cannot be resolved with sufficient confidence remain
unlinked.  ON DELETE SET NULL preserves track history if a Person record is
removed.

end_time is NULL while the track is still active; the pipeline fills it when
the person leaves the frame for good.
"""

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Optional

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.entities.event.model import Event
    from app.entities.observation.model import Observation
    from app.entities.person.model import Person
    from app.entities.state_interval.model import StateInterval
    from app.entities.video.model import Video


class Track(Base):
    __tablename__ = "tracks"

    __table_args__ = (
        # (video_id, local_track_id) is the natural key: tracker IDs restart
        # per-video.
        sa.UniqueConstraint("video_id", "local_track_id", name="uq_tracks_video_local"),
        # end_time must be at or after start_time when set.
        sa.CheckConstraint(
            "end_time IS NULL OR end_time >= start_time",
            name="ck_tracks_end_after_start",
        ),
        # Composite index for "all tracks for person P ordered by time".
        sa.Index("ix_tracks_person_start", "person_id", "start_time"),
        sa.CheckConstraint(
            "identity_confidence IS NULL OR identity_confidence BETWEEN 0.0 AND 1.0",
            name="ck_tracks_identity_confidence",
        ),
    )

    # ── Columns ───────────────────────────────────────────────────────────────
    id: Mapped[uuid.UUID] = mapped_column(
        sa.Uuid(as_uuid=True),
        primary_key=True,
        server_default=sa.text("gen_random_uuid()"),
        default=uuid.uuid4,
    )
    video_id: Mapped[uuid.UUID] = mapped_column(
        sa.Uuid(as_uuid=True),
        sa.ForeignKey("videos.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    local_track_id: Mapped[str] = mapped_column(sa.String(80), nullable=False)
    person_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        sa.Uuid(as_uuid=True),
        sa.ForeignKey("persons.id", ondelete="SET NULL"),
        nullable=True,
    )
    start_time: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), nullable=False
    )
    end_time: Mapped[Optional[datetime]] = mapped_column(
        sa.DateTime(timezone=True), nullable=True
    )

    # ── Identity decision (provisional – see migration b7e1c0d2f3a4) ──────────
    # Open string, no CHECK: values like probable/uncertain/new may change.
    identity_status: Mapped[Optional[str]] = mapped_column(sa.String(30), nullable=True)
    identity_confidence: Mapped[Optional[float]] = mapped_column(sa.Float, nullable=True)
    # Evolving evidence (candidates, scores, geometry verdict, model version …).
    identity_meta: Mapped[dict] = mapped_column(
        JSONB, nullable=False, server_default=sa.text("'{}'::jsonb"), default=dict
    )

    # ── Relationships ─────────────────────────────────────────────────────────
    video: Mapped["Video"] = relationship("Video", back_populates="tracks")
    person: Mapped[Optional["Person"]] = relationship(
        "Person", back_populates="tracks"
    )
    observations: Mapped[list["Observation"]] = relationship(
        "Observation", back_populates="track", passive_deletes=True
    )
    state_intervals: Mapped[list["StateInterval"]] = relationship(
        "StateInterval", back_populates="track", passive_deletes=True
    )
    # ON DELETE SET NULL: room-level events (entered_building …) can exist
    # even when the associated track is deleted.
    events: Mapped[list["Event"]] = relationship(
        "Event", back_populates="track", passive_deletes=True
    )
