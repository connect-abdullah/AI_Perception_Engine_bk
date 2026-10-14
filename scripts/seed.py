"""Seed script – anonymous dummy data for SQL query testing.

Run AFTER applying the schema:
    poetry run python scripts/seed.py

The script is idempotent on natural keys:
  - Skips rooms/cameras/videos/persons that already exist.
  - Always inserts a fresh track + observations + interval + event so you can
    re-run it without truncating tables.

NO real personal identities, face data, or video content is created.
"""

import os
import sys
import uuid
from datetime import datetime, timezone

# Allow running from the repo root without installing the package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv

load_dotenv()

from app.db.session import SessionLocal  # noqa: E402
from app.entities.camera.model import Camera  # noqa: E402
from app.entities.event.model import Event  # noqa: E402
from app.entities.observation.model import Observation  # noqa: E402
from app.entities.person.model import Person  # noqa: E402
from app.entities.room.model import Room  # noqa: E402
from app.entities.state_interval.model import StateInterval  # noqa: E402
from app.entities.track.model import Track  # noqa: E402
from app.entities.video.model import Video  # noqa: E402

# ── Helpers ───────────────────────────────────────────────────────────────────

NOW = datetime.now(tz=timezone.utc)


def utc(year=2026, month=10, day=14, hour=9, minute=0, second=0) -> datetime:
    return datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)


# ── Seed ──────────────────────────────────────────────────────────────────────

db = SessionLocal()

try:
    # ── Room ──────────────────────────────────────────────────────────────────
    room = db.query(Room).filter_by(name="Lobby").first()
    if not room:
        room = Room(name="Lobby", floor=0)
        db.add(room)
        db.flush()
        print(f"Created room: {room.id}  (Lobby, floor 0)")
    else:
        print(f"Room exists:  {room.id}  (Lobby)")

    # ── Camera ────────────────────────────────────────────────────────────────
    camera = db.query(Camera).filter_by(camera_code="CAM-LOBBY-01").first()
    if not camera:
        camera = Camera(camera_code="CAM-LOBBY-01", room_id=room.id)
        db.add(camera)
        db.flush()
        print(f"Created camera: {camera.id}  (CAM-LOBBY-01)")
    else:
        print(f"Camera exists:  {camera.id}  (CAM-LOBBY-01)")

    # ── Video ─────────────────────────────────────────────────────────────────
    video = (
        db.query(Video)
        .filter_by(file_path="samples/lobby_2026-10-14_090000.mp4")
        .first()
    )
    if not video:
        video = Video(
            camera_id=camera.id,
            file_path="samples/lobby_2026-10-14_090000.mp4",
            video_start_time=utc(hour=9),
            fps=25.0,
            width=1920,
            height=1080,
            duration_seconds=300.0,
            status="completed",
        )
        db.add(video)
        db.flush()
        print(f"Created video: {video.id}")
    else:
        print(f"Video exists:  {video.id}")

    # ── Person (anonymous) ────────────────────────────────────────────────────
    person = db.query(Person).filter_by(person_code="P001").first()
    if not person:
        person = Person(person_code="P001", known_name=None)
        db.add(person)
        db.flush()
        print(f"Created person: {person.id}  (P001, anonymous)")
    else:
        print(f"Person exists:  {person.id}  (P001)")

    # ── Track ─────────────────────────────────────────────────────────────────
    # Always create a fresh track so we can inspect observations each run.
    local_tid = f"T{uuid.uuid4().hex[:8]}"
    track = Track(
        video_id=video.id,
        local_track_id=local_tid,
        person_id=person.id,
        start_time=utc(hour=9, minute=1),
        end_time=utc(hour=9, minute=4),
    )
    db.add(track)
    db.flush()
    print(f"Created track:  {track.id}  (local={local_tid})")

    # ── Observations (3 sample frames) ───────────────────────────────────────
    for i, (frame, minute, second) in enumerate([(0, 1, 0), (25, 1, 1), (50, 1, 2)]):
        obs = Observation(
            track_id=track.id,
            frame_number=frame,
            timestamp=utc(hour=9, minute=minute, second=second),
            bbox_x=320,
            bbox_y=200,
            bbox_width=80,
            bbox_height=180,
            detection_confidence=round(0.92 - i * 0.03, 2),
            activity="walking",
            activity_confidence=round(0.85 - i * 0.05, 2),
        )
        db.add(obs)
    db.flush()
    print("Created 3 observations (frames 0, 25, 50)")

    # ── State interval ────────────────────────────────────────────────────────
    interval = StateInterval(
        track_id=track.id,
        activity="walking",
        start_time=utc(hour=9, minute=1),
        end_time=utc(hour=9, minute=4),
        confidence=0.88,
    )
    db.add(interval)
    db.flush()
    print(f"Created state interval: walking  {interval.id}")

    # ── Event ─────────────────────────────────────────────────────────────────
    event = Event(
        track_id=track.id,
        event_type="entered_building",
        timestamp=utc(hour=9, minute=1),
        room_id=room.id,
        confidence=0.95,
        metadata={"entry_door": "main"},
    )
    db.add(event)
    db.flush()
    print(f"Created event: entered_building  {event.id}")

    db.commit()
    print("\nSeed complete.")

except Exception as exc:
    db.rollback()
    print(f"\nSeed FAILED: {exc}", file=sys.stderr)
    sys.exit(1)
finally:
    db.close()
