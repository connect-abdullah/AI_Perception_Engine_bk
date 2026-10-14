"""Declarative base – import ALL entity models here so Alembic can discover
them through a single metadata object.  Entity models themselves import Base
from this module to avoid circular imports.
"""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


# ── Entity model registration (ORDER matters: parents before children) ────────
from app.entities.room.model import Room  # noqa: F401, E402
from app.entities.camera.model import Camera  # noqa: F401, E402
from app.entities.video.model import Video  # noqa: F401, E402
from app.entities.person.model import Person  # noqa: F401, E402
from app.entities.track.model import Track  # noqa: F401, E402
from app.entities.observation.model import Observation  # noqa: F401, E402
from app.entities.state_interval.model import StateInterval  # noqa: F401, E402
from app.entities.event.model import Event  # noqa: F401, E402
