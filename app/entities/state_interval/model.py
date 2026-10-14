"""StateInterval model.

A state interval summarises a continuous activity for a track over a time
window.  While observations represent individual detection frames, a state
interval records that a person was "standing" from T1 to T2.

activity is kept as an open VARCHAR(30): the CV pipeline can emit new labels
without requiring a schema migration.  Expected values for this PoC are
standing, sitting, walking, running, lying.

end_time is NULL while the activity is still ongoing.

confidence is optional (NULL if the activity classifier did not produce one)
and constrained to [0, 1].
"""

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Optional

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.entities.track.model import Track


class StateInterval(Base):
    __tablename__ = "state_intervals"

    __table_args__ = (
        sa.CheckConstraint(
            "end_time IS NULL OR end_time >= start_time",
            name="ck_si_end_after_start",
        ),
        sa.CheckConstraint(
            "confidence IS NULL OR confidence BETWEEN 0.0 AND 1.0",
            name="ck_si_confidence",
        ),
        # Supports "all intervals for track T ordered by start_time".
        sa.Index("ix_si_track_start", "track_id", "start_time"),
    )

    # ── Columns ───────────────────────────────────────────────────────────────
    id: Mapped[uuid.UUID] = mapped_column(
        sa.Uuid(as_uuid=True),
        primary_key=True,
        server_default=sa.text("gen_random_uuid()"),
        default=uuid.uuid4,
    )
    track_id: Mapped[uuid.UUID] = mapped_column(
        sa.Uuid(as_uuid=True),
        sa.ForeignKey("tracks.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    activity: Mapped[str] = mapped_column(sa.String(30), nullable=False)
    start_time: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), nullable=False
    )
    end_time: Mapped[Optional[datetime]] = mapped_column(
        sa.DateTime(timezone=True), nullable=True
    )
    confidence: Mapped[Optional[float]] = mapped_column(sa.Float, nullable=True)

    # ── Relationships ─────────────────────────────────────────────────────────
    track: Mapped["Track"] = relationship("Track", back_populates="state_intervals")
