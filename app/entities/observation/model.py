"""Observation model.

An observation is one detection snapshot: the bounding box and activity label
for a track at a specific frame.  The pipeline controls sampling frequency;
not every frame needs an observation row.

id is a BIGINT GENERATED ALWAYS AS IDENTITY – observations are write-once,
high-volume rows that do not need a UUID primary key.

(track_id, frame_number) is unique: a track can only appear once per frame.

bbox_x / bbox_y are unconstrained so bounding boxes that touch the frame edge
can have x=0 or y=0.  Width and height must be strictly positive.

Both confidence values are in [0, 1]; activity_confidence is nullable because
not every detection includes an activity classifier result.
"""

from datetime import datetime
from typing import TYPE_CHECKING, Optional

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.entities.track.model import Track

import uuid


class Observation(Base):
    __tablename__ = "observations"

    __table_args__ = (
        # Prevents duplicate observations for the same frame within a track.
        sa.UniqueConstraint(
            "track_id", "frame_number", name="uq_observations_track_frame"
        ),
        # Domain constraints validated at the database level.
        sa.CheckConstraint("frame_number >= 0", name="ck_obs_frame_non_negative"),
        sa.CheckConstraint("bbox_width > 0", name="ck_obs_bbox_width"),
        sa.CheckConstraint("bbox_height > 0", name="ck_obs_bbox_height"),
        sa.CheckConstraint(
            "detection_confidence BETWEEN 0.0 AND 1.0",
            name="ck_obs_det_confidence",
        ),
        sa.CheckConstraint(
            "activity_confidence IS NULL OR activity_confidence BETWEEN 0.0 AND 1.0",
            name="ck_obs_act_confidence",
        ),
        # Supports "all observations for track T in time window [a, b]".
        sa.Index("ix_observations_track_ts", "track_id", "timestamp"),
        # Supports "who was in room R between T1 and T2".
        sa.Index("ix_observations_room_ts", "room_id", "timestamp"),
    )

    # ── Columns ───────────────────────────────────────────────────────────────
    # BIGINT GENERATED ALWAYS AS IDENTITY – compact, sequential, write-once.
    id: Mapped[int] = mapped_column(
        sa.BigInteger, sa.Identity(always=True), primary_key=True
    )
    track_id: Mapped[uuid.UUID] = mapped_column(
        sa.Uuid(as_uuid=True),
        sa.ForeignKey("tracks.id", ondelete="CASCADE"),
        nullable=False,
    )
    frame_number: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), nullable=False
    )
    bbox_x: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    bbox_y: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    bbox_width: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    bbox_height: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    detection_confidence: Mapped[float] = mapped_column(sa.Float, nullable=False)
    activity: Mapped[Optional[str]] = mapped_column(sa.String(30), nullable=True)
    activity_confidence: Mapped[Optional[float]] = mapped_column(
        sa.Float, nullable=True
    )
    # Optional world / floor-plan position – filled only when the spatial layer
    # can compute it (see migration b7e1c0d2f3a4).
    world_x: Mapped[Optional[float]] = mapped_column(sa.Float, nullable=True)
    world_y: Mapped[Optional[float]] = mapped_column(sa.Float, nullable=True)
    world_z: Mapped[Optional[float]] = mapped_column(sa.Float, nullable=True)
    room_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        sa.Uuid(as_uuid=True),
        sa.ForeignKey("rooms.id", ondelete="SET NULL", name="fk_observations_room"),
        nullable=True,
    )

    # ── Relationships ─────────────────────────────────────────────────────────
    track: Mapped["Track"] = relationship("Track", back_populates="observations")
