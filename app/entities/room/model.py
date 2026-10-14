"""Room model.

A room represents a physical space in the building.  Cameras and security
events are optionally associated with a room; the association is nullable so
that cameras can be registered before floor-plans are finalised.
"""

import uuid
from typing import TYPE_CHECKING, Optional

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.entities.camera.model import Camera
    from app.entities.event.model import Event


class Room(Base):
    __tablename__ = "rooms"

    # ── Columns ───────────────────────────────────────────────────────────────
    id: Mapped[uuid.UUID] = mapped_column(
        sa.Uuid(as_uuid=True),
        primary_key=True,
        # gen_random_uuid() is provided by the pgcrypto extension, which
        # Supabase enables by default.
        server_default=sa.text("gen_random_uuid()"),
        default=uuid.uuid4,
    )
    name: Mapped[str] = mapped_column(sa.String(120), nullable=False)
    # NULL floor means the floor is unknown or not yet assigned.
    floor: Mapped[Optional[int]] = mapped_column(sa.Integer, nullable=True)

    # ── Relationships ─────────────────────────────────────────────────────────
    # ON DELETE SET NULL in FK → cameras: a room can be deleted without losing
    # camera records.  The back-reference here is passive (no cascade).
    cameras: Mapped[list["Camera"]] = relationship(
        "Camera", back_populates="room", passive_deletes=True
    )
    events: Mapped[list["Event"]] = relationship(
        "Event", back_populates="room", passive_deletes=True
    )
