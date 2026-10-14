"""Camera model.

camera_code is a short human-readable identifier (e.g. CAM-01, LOBBY-N).
It is unique across the system so external integrations can refer to a camera
without knowing the internal UUID.

room_id is nullable: a camera can be registered before it is physically
assigned to a room.  ON DELETE SET NULL preserves camera records when a room
is removed.
"""

import uuid
from typing import TYPE_CHECKING, Optional

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.entities.room.model import Room
    from app.entities.video.model import Video


class Camera(Base):
    __tablename__ = "cameras"

    # ── Columns ───────────────────────────────────────────────────────────────
    id: Mapped[uuid.UUID] = mapped_column(
        sa.Uuid(as_uuid=True),
        primary_key=True,
        server_default=sa.text("gen_random_uuid()"),
        default=uuid.uuid4,
    )
    camera_code: Mapped[str] = mapped_column(sa.String(50), unique=True, nullable=False)
    room_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        sa.Uuid(as_uuid=True),
        sa.ForeignKey("rooms.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    # ── Relationships ─────────────────────────────────────────────────────────
    room: Mapped[Optional["Room"]] = relationship("Room", back_populates="cameras")
    # ON DELETE RESTRICT in FK → videos: removing a camera is blocked while
    # video metadata records exist, preventing silent data loss.
    videos: Mapped[list["Video"]] = relationship(
        "Video", back_populates="camera", passive_deletes=True
    )
