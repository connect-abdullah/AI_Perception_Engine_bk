"""Person model.

A Person represents a global, anonymous identity that spans multiple camera
views and recording sessions.  The ReID component assigns a local track to a
person with sufficient confidence; uncertain matches leave track.person_id NULL.

person_code is a human-readable shortcode (P001, P002 …) unique across the
system.  It is used in natural-language queries ("show all events for P001")
and in reports.

known_name is intentionally nullable: the system is anonymous by design.
A name is only filled when the operator explicitly links an identity.
"""

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Optional

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.entities.track.model import Track


class Person(Base):
    __tablename__ = "persons"

    # ── Columns ───────────────────────────────────────────────────────────────
    id: Mapped[uuid.UUID] = mapped_column(
        sa.Uuid(as_uuid=True),
        primary_key=True,
        server_default=sa.text("gen_random_uuid()"),
        default=uuid.uuid4,
    )
    person_code: Mapped[str] = mapped_column(
        sa.String(20), unique=True, nullable=False
    )
    known_name: Mapped[Optional[str]] = mapped_column(sa.String(120), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), server_default=sa.func.now()
    )

    # ── Relationships ─────────────────────────────────────────────────────────
    # ON DELETE SET NULL in FK → tracks: removing a person identity record
    # leaves track.person_id NULL so track history is preserved.
    tracks: Mapped[list["Track"]] = relationship(
        "Track", back_populates="person", passive_deletes=True
    )
