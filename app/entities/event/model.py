"""Event model.

An event captures a meaningful security-relevant occurrence at a point in time.
Unlike observations (individual frames) or state intervals (activity windows),
events are discrete facts: a person entering a room, a floor change, etc.

Expected event_type values for this PoC:
    entered_building, exited_building, entered_room, left_room, floor_changed

event_type is an open VARCHAR(50) so new event kinds can be added by the
pipeline without a schema migration.

track_id and room_id are both nullable:
  - track_id: an event may outlive its track or may be camera-level only.
    ON DELETE SET NULL preserves the event.
  - room_id: building-level events (entered_building) have no specific room.
    ON DELETE SET NULL preserves the event.

metadata (JSONB) stores event-specific extras that do not warrant their own
column (e.g. direction, floor_from/floor_to for floor_changed).  It defaults
to an empty object and is never NULL so callers can always merge into it.
"""

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Optional

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.entities.room.model import Room
    from app.entities.track.model import Track


class Event(Base):
    __tablename__ = "events"

    __table_args__ = (
        sa.CheckConstraint(
            "confidence IS NULL OR confidence BETWEEN 0.0 AND 1.0",
            name="ck_events_confidence",
        ),
        # "All events in room R ordered by time" – the most common security query.
        sa.Index("ix_events_room_ts", "room_id", "timestamp"),
        # "All events for track T" – needed when building a person timeline.
        sa.Index("ix_events_track_id", "track_id"),
        # "All events of type X in time range" – alert / report queries.
        sa.Index("ix_events_type_ts", "event_type", "timestamp"),
    )

    # ── Columns ───────────────────────────────────────────────────────────────
    id: Mapped[uuid.UUID] = mapped_column(
        sa.Uuid(as_uuid=True),
        primary_key=True,
        server_default=sa.text("gen_random_uuid()"),
        default=uuid.uuid4,
    )
    track_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        sa.Uuid(as_uuid=True),
        sa.ForeignKey("tracks.id", ondelete="SET NULL"),
        nullable=True,
    )
    event_type: Mapped[str] = mapped_column(sa.String(50), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), nullable=False
    )
    room_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        sa.Uuid(as_uuid=True),
        sa.ForeignKey("rooms.id", ondelete="SET NULL"),
        nullable=True,
    )
    confidence: Mapped[Optional[float]] = mapped_column(sa.Float, nullable=True)
    # JSONB NOT NULL DEFAULT '{}' – callers can always json-merge; never null.
    # Column name in the database is "metadata"; the Python attribute is
    # "event_meta" to avoid shadowing DeclarativeBase.metadata.
    event_meta: Mapped[dict] = mapped_column(
        "metadata", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")
    )

    # ── Relationships ─────────────────────────────────────────────────────────
    track: Mapped[Optional["Track"]] = relationship("Track", back_populates="events")
    room: Mapped[Optional["Room"]] = relationship("Room", back_populates="events")
