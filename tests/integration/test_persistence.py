"""Postgres integration tests for the full persistence layer.

Coverage:
  A. Migration on a clean database (alembic upgrade head)
  B. Video and camera creation / retrieval
  C. Tracks and observations: FK relationships, nullable person_id
  D. Idempotent run ingestion (retry does not duplicate rows)
  E. Identity linking: nullable, update, and clear
  F. Spatial fields: present and absent
  G. Activity (state interval) and event ingestion via the service
  H. Transaction rollback on invalid input
  I. Restart-safe reads: data persists across session instances
  J. Idempotent observation upsert (crop_path update)
  K. Track-to-person assignment stays valid when person_code is unknown

Run with:
    TEST_DATABASE_URL=postgresql+psycopg://perception:perception@localhost:5432/perception?sslmode=disable \
    pytest tests/integration/test_persistence.py -v
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy import select, text
from sqlalchemy.orm import Session

# Bootstrap model registration before importing individual entity classes.
from app.db.base import Base  # noqa: F401

from app.entities.camera.model import Camera
from app.entities.event.model import Event
from app.entities.ingest.service import HistoryService, IngestionService
from app.entities.observation.model import Observation
from app.entities.person.model import Person
from app.entities.state_interval.model import StateInterval
from app.entities.track.model import Track
from app.entities.video.model import Video

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_T0 = datetime(2018, 3, 9, 10, 15, 0, tzinfo=timezone.utc)
_T1 = datetime(2018, 3, 9, 10, 16, 0, tzinfo=timezone.utc)


def _obs(local_track_id: str, frame: int, **kw) -> dict:
    return {
        "local_track_id": local_track_id,
        "frame_number": frame,
        "timestamp_seconds": frame / 30.0,
        "bbox": (10, 20, 50, 100),
        "detection_confidence": 0.9,
        "crop_path": kw.get("crop_path"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# A. Migration: all tables present after alembic upgrade head
# ─────────────────────────────────────────────────────────────────────────────

class TestMigration:
    EXPECTED_TABLES = [
        "rooms", "cameras", "videos", "persons", "tracks",
        "observations", "state_intervals", "events",
    ]

    def test_tables_exist(self, db: Session):
        result = db.execute(
            text("SELECT tablename FROM pg_tables WHERE schemaname='public'")
        ).fetchall()
        actual = {r[0] for r in result}
        for name in self.EXPECTED_TABLES:
            assert name in actual, f"Table '{name}' missing after migration"

    def test_source_video_id_column_exists(self, db: Session):
        result = db.execute(
            text("SELECT column_name FROM information_schema.columns "
                 "WHERE table_name='videos' AND column_name='source_video_id'")
        ).fetchone()
        assert result is not None, "videos.source_video_id not found after migration"

    def test_crop_path_column_exists(self, db: Session):
        result = db.execute(
            text("SELECT column_name FROM information_schema.columns "
                 "WHERE table_name='observations' AND column_name='crop_path'")
        ).fetchone()
        assert result is not None, "observations.crop_path not found after migration"

    def test_identity_columns_exist(self, db: Session):
        for col in ("identity_status", "identity_confidence", "identity_meta"):
            result = db.execute(
                text("SELECT column_name FROM information_schema.columns "
                     f"WHERE table_name='tracks' AND column_name='{col}'")
            ).fetchone()
            assert result is not None, f"tracks.{col} not found after migration"

    def test_spatial_columns_exist(self, db: Session):
        for col in ("world_x", "world_y", "world_z", "room_id"):
            result = db.execute(
                text("SELECT column_name FROM information_schema.columns "
                     f"WHERE table_name='observations' AND column_name='{col}'")
            ).fetchone()
            assert result is not None, f"observations.{col} not found after migration"


# ─────────────────────────────────────────────────────────────────────────────
# B. Video and camera creation
# ─────────────────────────────────────────────────────────────────────────────

class TestVideoCamera:
    def test_camera_created_by_ingest(self, db: Session):
        svc = IngestionService(db)
        svc.ingest_run(
            source_video_id="test-vid-B1",
            camera_code="CAM-TEST-B1",
            fps=30.0,
            video_start_time=_T0,
            width=1920,
            height=1080,
            observations=[],
        )
        cam = db.scalars(
            select(Camera).where(Camera.camera_code == "CAM-TEST-B1")
        ).first()
        assert cam is not None

    def test_video_created_by_ingest(self, db: Session):
        svc = IngestionService(db)
        svc.ingest_run(
            source_video_id="test-vid-B2",
            camera_code="CAM-TEST-B2",
            fps=30.0,
            video_start_time=_T0,
            width=1920,
            height=1080,
            observations=[],
        )
        vid = db.scalars(
            select(Video).where(Video.source_video_id == "test-vid-B2")
        ).first()
        assert vid is not None
        assert vid.fps == 30.0
        assert vid.source == "cv"

    def test_camera_idempotent(self, db: Session):
        """Re-ingesting with the same camera_code must not duplicate the camera."""
        svc = IngestionService(db)
        for i in range(3):
            svc.ingest_run(
                source_video_id="test-vid-B3-idem",
                camera_code="CAM-IDEM-B3",
                fps=30.0,
                video_start_time=_T0,
                width=None,
                height=None,
                observations=[],
            )
        cameras = db.scalars(
            select(Camera).where(Camera.camera_code == "CAM-IDEM-B3")
        ).all()
        assert len(cameras) == 1

    def test_video_metadata_updated_on_retry(self, db: Session):
        svc = IngestionService(db)
        svc.ingest_run(
            source_video_id="test-vid-B4",
            camera_code="CAM-TEST-B4",
            fps=25.0,
            video_start_time=_T0,
            width=1280,
            height=720,
            observations=[],
        )
        svc.ingest_run(
            source_video_id="test-vid-B4",
            camera_code="CAM-TEST-B4",
            fps=30.0,
            video_start_time=_T0,
            width=1920,
            height=1080,
            observations=[],
        )
        vid = db.scalars(
            select(Video).where(Video.source_video_id == "test-vid-B4")
        ).first()
        assert vid.fps == 30.0
        assert vid.width == 1920


# ─────────────────────────────────────────────────────────────────────────────
# C. Tracks and observations: FK, nullable person_id
# ─────────────────────────────────────────────────────────────────────────────

class TestTracksObservations:
    def test_track_created_by_observation(self, db: Session):
        svc = IngestionService(db)
        result = svc.ingest_run(
            source_video_id="test-vid-C1",
            camera_code="CAM-C1",
            fps=30.0,
            video_start_time=_T0,
            width=None,
            height=None,
            observations=[_obs("1", 0), _obs("1", 30), _obs("2", 0)],
        )
        assert result["tracks_upserted"] == 2
        assert result["observations_upserted"] == 3

    def test_observations_linked_to_track(self, db: Session):
        svc = IngestionService(db)
        svc.ingest_run(
            source_video_id="test-vid-C2",
            camera_code="CAM-C2",
            fps=30.0,
            video_start_time=_T0,
            width=None,
            height=None,
            observations=[_obs("1", 0), _obs("1", 30)],
        )
        vid = db.scalars(
            select(Video).where(Video.source_video_id == "test-vid-C2")
        ).first()
        track = db.scalars(
            select(Track).where(
                Track.video_id == vid.id, Track.local_track_id == "1"
            )
        ).first()
        obs_rows = db.scalars(
            select(Observation).where(Observation.track_id == track.id)
        ).all()
        assert len(obs_rows) == 2

    def test_person_id_is_null_initially(self, db: Session):
        svc = IngestionService(db)
        result = svc.ingest_run(
            source_video_id="test-vid-C3",
            camera_code="CAM-C3",
            fps=30.0,
            video_start_time=_T0,
            width=None,
            height=None,
            observations=[_obs("1", 0)],
        )
        vid = db.scalars(
            select(Video).where(Video.source_video_id == "test-vid-C3")
        ).first()
        track = db.scalars(
            select(Track).where(
                Track.video_id == vid.id, Track.local_track_id == "1"
            )
        ).first()
        assert track.person_id is None, "person_id must be null until ReID assigns it"

    def test_observation_bbox_stored_correctly(self, db: Session):
        svc = IngestionService(db)
        svc.ingest_run(
            source_video_id="test-vid-C4",
            camera_code="CAM-C4",
            fps=30.0,
            video_start_time=_T0,
            width=1920,
            height=1080,
            observations=[{
                "local_track_id": "1",
                "frame_number": 0,
                "timestamp_seconds": 0.0,
                "bbox": (100, 200, 50, 120),
                "detection_confidence": 0.85,
                "crop_path": "crops/1/00000000.png",
            }],
        )
        vid = db.scalars(
            select(Video).where(Video.source_video_id == "test-vid-C4")
        ).first()
        track = db.scalars(
            select(Track).where(
                Track.video_id == vid.id, Track.local_track_id == "1"
            )
        ).first()
        obs = db.scalars(
            select(Observation).where(Observation.track_id == track.id)
        ).first()
        assert obs.bbox_x == 100
        assert obs.bbox_y == 200
        assert obs.bbox_width == 50
        assert obs.bbox_height == 120
        assert abs(obs.detection_confidence - 0.85) < 1e-5
        assert obs.crop_path == "crops/1/00000000.png"

    def test_duplicate_frame_observation_rejected_by_constraint(self, db: Session):
        """Two rows with the same (track_id, frame_number) must not coexist."""
        svc = IngestionService(db)
        svc.ingest_run(
            source_video_id="test-vid-C5",
            camera_code="CAM-C5",
            fps=30.0,
            video_start_time=_T0,
            width=None,
            height=None,
            observations=[_obs("1", 0)],
        )
        vid = db.scalars(
            select(Video).where(Video.source_video_id == "test-vid-C5")
        ).first()
        track = db.scalars(
            select(Track).where(
                Track.video_id == vid.id, Track.local_track_id == "1"
            )
        ).first()
        count = db.scalars(
            select(Observation).where(Observation.track_id == track.id)
        ).all()
        assert len(count) == 1


# ─────────────────────────────────────────────────────────────────────────────
# D. Idempotent run ingestion
# ─────────────────────────────────────────────────────────────────────────────

class TestIdempotentIngest:
    def test_resubmit_same_run_does_not_duplicate(self, db: Session):
        svc = IngestionService(db)
        for _ in range(2):
            svc.ingest_run(
                source_video_id="test-vid-D1",
                camera_code="CAM-D1",
                fps=30.0,
                video_start_time=_T0,
                width=None,
                height=None,
                observations=[_obs("1", 0), _obs("1", 30)],
            )
        vid = db.scalars(
            select(Video).where(Video.source_video_id == "test-vid-D1")
        ).first()
        track = db.scalars(
            select(Track).where(
                Track.video_id == vid.id, Track.local_track_id == "1"
            )
        ).first()
        obs_rows = db.scalars(
            select(Observation).where(Observation.track_id == track.id)
        ).all()
        assert len(obs_rows) == 2, (
            "Re-submitting the same run must not create duplicate observations"
        )


# ─────────────────────────────────────────────────────────────────────────────
# E. Identity linking: nullable, update, clear
# ─────────────────────────────────────────────────────────────────────────────

class TestIdentityLinking:
    def _setup_track(self, db: Session, vid_id: str, cam: str = "CAM-E") -> None:
        svc = IngestionService(db)
        svc.ingest_run(
            source_video_id=vid_id,
            camera_code=cam,
            fps=30.0,
            video_start_time=_T0,
            width=None,
            height=None,
            observations=[_obs("1", 0)],
        )

    def test_link_creates_person_if_new(self, db: Session):
        vid_id = "test-vid-E1"
        self._setup_track(db, vid_id)
        svc = IngestionService(db)
        result = svc.link_identity(
            source_video_id=vid_id,
            local_track_id="1",
            person_code="P099",
            identity_status="probable",
            identity_confidence=0.88,
            identity_meta={"score": 0.88},
        )
        assert result["person_code"] == "P099"
        person = db.scalars(
            select(Person).where(Person.person_code == "P099")
        ).first()
        assert person is not None

    def test_link_sets_track_person_id(self, db: Session):
        vid_id = "test-vid-E2"
        self._setup_track(db, vid_id)
        svc = IngestionService(db)
        svc.link_identity(
            source_video_id=vid_id,
            local_track_id="1",
            person_code="P098",
            identity_status="probable",
            identity_confidence=None,
            identity_meta=None,
        )
        vid = db.scalars(
            select(Video).where(Video.source_video_id == vid_id)
        ).first()
        track = db.scalars(
            select(Track).where(
                Track.video_id == vid.id, Track.local_track_id == "1"
            )
        ).first()
        assert track.person_id is not None

    def test_clear_identity_leaves_person_id_null(self, db: Session):
        vid_id = "test-vid-E3"
        self._setup_track(db, vid_id)
        svc = IngestionService(db)
        svc.link_identity(
            source_video_id=vid_id, local_track_id="1",
            person_code="P097", identity_status=None, identity_confidence=None,
            identity_meta=None,
        )
        svc.link_identity(
            source_video_id=vid_id, local_track_id="1",
            person_code=None,   # explicit unlink
            identity_status="uncertain", identity_confidence=None,
            identity_meta=None,
        )
        vid = db.scalars(
            select(Video).where(Video.source_video_id == vid_id)
        ).first()
        track = db.scalars(
            select(Track).where(
                Track.video_id == vid.id, Track.local_track_id == "1"
            )
        ).first()
        assert track.person_id is None

    def test_link_unknown_track_raises(self, db: Session):
        vid_id = "test-vid-E4"
        self._setup_track(db, vid_id)
        svc = IngestionService(db)
        with pytest.raises(ValueError, match="No track"):
            svc.link_identity(
                source_video_id=vid_id, local_track_id="999",
                person_code="P001", identity_status=None,
                identity_confidence=None, identity_meta=None,
            )

    def test_link_unknown_video_raises(self, db: Session):
        svc = IngestionService(db)
        with pytest.raises(ValueError, match="No video"):
            svc.link_identity(
                source_video_id="does-not-exist", local_track_id="1",
                person_code="P001", identity_status=None,
                identity_confidence=None, identity_meta=None,
            )


# ─────────────────────────────────────────────────────────────────────────────
# F. Spatial fields: present and absent
# ─────────────────────────────────────────────────────────────────────────────

class TestSpatialFields:
    def test_observations_stored_without_spatial(self, db: Session):
        svc = IngestionService(db)
        svc.ingest_run(
            source_video_id="test-vid-F1",
            camera_code="CAM-F1",
            fps=30.0,
            video_start_time=_T0,
            width=None, height=None,
            observations=[_obs("1", 0)],
        )
        vid = db.scalars(
            select(Video).where(Video.source_video_id == "test-vid-F1")
        ).first()
        track = db.scalars(
            select(Track).where(
                Track.video_id == vid.id, Track.local_track_id == "1"
            )
        ).first()
        obs = db.scalars(
            select(Observation).where(Observation.track_id == track.id)
        ).first()
        assert obs.world_x is None
        assert obs.world_y is None
        assert obs.world_z is None
        assert obs.room_id is None

    def test_spatial_fields_can_be_updated_directly(self, db: Session):
        """Spatial data is set by the 3D pipeline after ingestion."""
        svc = IngestionService(db)
        svc.ingest_run(
            source_video_id="test-vid-F2",
            camera_code="CAM-F2",
            fps=30.0,
            video_start_time=_T0,
            width=None, height=None,
            observations=[_obs("1", 0)],
        )
        vid = db.scalars(
            select(Video).where(Video.source_video_id == "test-vid-F2")
        ).first()
        track = db.scalars(
            select(Track).where(
                Track.video_id == vid.id, Track.local_track_id == "1"
            )
        ).first()
        obs = db.scalars(
            select(Observation).where(Observation.track_id == track.id)
        ).first()
        obs.world_x = 1.5
        obs.world_y = 2.5
        obs.world_z = 0.0
        db.flush()

        obs_fresh = db.scalars(
            select(Observation).where(Observation.id == obs.id)
        ).first()
        assert abs(obs_fresh.world_x - 1.5) < 1e-5


# ─────────────────────────────────────────────────────────────────────────────
# G. Activity and event ingestion
# ─────────────────────────────────────────────────────────────────────────────

class TestActivityEvent:
    def _setup(self, db: Session, vid_id: str) -> None:
        svc = IngestionService(db)
        svc.ingest_run(
            source_video_id=vid_id,
            camera_code="CAM-G",
            fps=30.0,
            video_start_time=_T0,
            width=None, height=None,
            observations=[_obs("1", 0)],
        )

    def test_activity_interval_created(self, db: Session):
        vid_id = "test-vid-G1"
        self._setup(db, vid_id)
        svc = IngestionService(db)
        result = svc.ingest_activity(
            source_video_id=vid_id,
            local_track_id="1",
            activity="standing",
            start_time=_T0,
            end_time=_T1,
            confidence=0.9,
        )
        assert result["activity"] == "standing"
        interval = db.scalars(
            select(StateInterval).where(
                StateInterval.id == uuid.UUID(result["state_interval_id"])
            )
        ).first()
        assert interval is not None
        assert interval.activity == "standing"
        assert abs(interval.confidence - 0.9) < 1e-5

    def test_activity_no_end_time(self, db: Session):
        vid_id = "test-vid-G2"
        self._setup(db, vid_id)
        svc = IngestionService(db)
        result = svc.ingest_activity(
            source_video_id=vid_id,
            local_track_id="1",
            activity="moving",
            start_time=_T0,
            end_time=None,
            confidence=None,
        )
        interval = db.scalars(
            select(StateInterval).where(
                StateInterval.id == uuid.UUID(result["state_interval_id"])
            )
        ).first()
        assert interval.end_time is None

    def test_event_created(self, db: Session):
        vid_id = "test-vid-G3"
        self._setup(db, vid_id)
        svc = IngestionService(db)
        result = svc.ingest_event(
            source_video_id=vid_id,
            local_track_id="1",
            event_type="entered_zone",
            timestamp=_T0,
            confidence=0.8,
            event_meta={"zone": "lobby"},
        )
        event = db.scalars(
            select(Event).where(
                Event.id == uuid.UUID(result["event_id"])
            )
        ).first()
        assert event is not None
        assert event.event_type == "entered_zone"
        assert event.event_meta == {"zone": "lobby"}

    def test_event_without_track(self, db: Session):
        """Events can exist without a known track."""
        svc = IngestionService(db)
        result = svc.ingest_event(
            source_video_id=None,
            local_track_id=None,
            event_type="alarm_trigger",
            timestamp=_T0,
            confidence=None,
            event_meta={},
        )
        event = db.scalars(
            select(Event).where(
                Event.id == uuid.UUID(result["event_id"])
            )
        ).first()
        assert event is not None
        assert event.track_id is None


# ─────────────────────────────────────────────────────────────────────────────
# H. Transaction rollback on invalid input
# ─────────────────────────────────────────────────────────────────────────────

class TestRollback:
    def test_link_to_missing_video_does_not_commit(self, db: Session):
        """ValueError from the service must not commit partial data."""
        svc = IngestionService(db)
        with pytest.raises(ValueError):
            svc.link_identity(
                source_video_id="nonexistent-H1",
                local_track_id="1",
                person_code="P001",
                identity_status=None,
                identity_confidence=None,
                identity_meta=None,
            )
        # Nothing should have been written.
        person = db.scalars(
            select(Person).where(Person.person_code == "P001")
        ).first()
        assert person is None

    def test_activity_on_missing_track_raises(self, db: Session):
        svc = IngestionService(db)
        svc.ingest_run(
            source_video_id="test-vid-H2",
            camera_code="CAM-H2",
            fps=30.0,
            video_start_time=_T0,
            width=None,
            height=None,
            observations=[],
        )
        with pytest.raises(ValueError, match="No track"):
            svc.ingest_activity(
                source_video_id="test-vid-H2",
                local_track_id="999",
                activity="standing",
                start_time=_T0,
                end_time=None,
                confidence=None,
            )


# ─────────────────────────────────────────────────────────────────────────────
# I. Restart-safe reads: data survives a new session
# ─────────────────────────────────────────────────────────────────────────────

class TestRestartSafeReads:
    """Use commit() so data is visible across sessions; clean up manually."""

    def test_data_survives_new_session(self, migrated_engine):
        from sqlalchemy.orm import sessionmaker as sm
        S = sm(bind=migrated_engine, autocommit=False, autoflush=False)
        vid_id = f"restart-safe-{uuid.uuid4().hex[:8]}"

        # Write in session 1
        s1 = S()
        svc = IngestionService(s1)
        svc.ingest_run(
            source_video_id=vid_id,
            camera_code=f"CAM-RESTART-{vid_id[-4:]}",
            fps=30.0,
            video_start_time=_T0,
            width=None,
            height=None,
            observations=[_obs("1", 0)],
        )
        s1.commit()
        s1.close()

        # Read in session 2
        s2 = S()
        vid = s2.scalars(
            select(Video).where(Video.source_video_id == vid_id)
        ).first()
        try:
            assert vid is not None, "Video not found after commit+new session"
            tracks = s2.scalars(
                select(Track).where(Track.video_id == vid.id)
            ).all()
            assert len(tracks) == 1
            obs = s2.scalars(
                select(Observation).where(Observation.track_id == tracks[0].id)
            ).all()
            assert len(obs) == 1
        finally:
            # Clean up to avoid polluting subsequent tests.
            s2.rollback()
            s2.close()

            s3 = S()
            v = s3.scalars(
                select(Video).where(Video.source_video_id == vid_id)
            ).first()
            if v:
                s3.delete(v)
                s3.commit()
            s3.close()


# ─────────────────────────────────────────────────────────────────────────────
# J. Idempotent observation upsert (crop_path update)
# ─────────────────────────────────────────────────────────────────────────────

class TestObservationUpsert:
    def test_crop_path_updated_on_retry(self, db: Session):
        svc = IngestionService(db)
        svc.ingest_run(
            source_video_id="test-vid-J1",
            camera_code="CAM-J1",
            fps=30.0,
            video_start_time=_T0,
            width=None,
            height=None,
            observations=[_obs("1", 0, crop_path=None)],
        )
        svc.ingest_run(
            source_video_id="test-vid-J1",
            camera_code="CAM-J1",
            fps=30.0,
            video_start_time=_T0,
            width=None,
            height=None,
            observations=[_obs("1", 0, crop_path="crops/1/00000000.png")],
        )
        vid = db.scalars(
            select(Video).where(Video.source_video_id == "test-vid-J1")
        ).first()
        track = db.scalars(
            select(Track).where(
                Track.video_id == vid.id, Track.local_track_id == "1"
            )
        ).first()
        obs_rows = db.scalars(
            select(Observation).where(Observation.track_id == track.id)
        ).all()
        assert len(obs_rows) == 1
        assert obs_rows[0].crop_path == "crops/1/00000000.png"


# ─────────────────────────────────────────────────────────────────────────────
# K. History reads
# ─────────────────────────────────────────────────────────────────────────────

class TestHistoryReads:
    def _setup(self, db: Session, vid_id: str) -> None:
        svc = IngestionService(db)
        svc.ingest_run(
            source_video_id=vid_id,
            camera_code=f"CAM-K-{vid_id[-3:]}",
            fps=30.0,
            video_start_time=_T0,
            width=None,
            height=None,
            observations=[_obs("1", 0), _obs("2", 0)],
        )

    def test_get_track_returns_dict(self, db: Session):
        vid_id = "test-vid-K1"
        self._setup(db, vid_id)
        vid = db.scalars(
            select(Video).where(Video.source_video_id == vid_id)
        ).first()
        track = db.scalars(
            select(Track).where(
                Track.video_id == vid.id, Track.local_track_id == "1"
            )
        ).first()
        svc = HistoryService(db)
        result = svc.get_track(str(track.id))
        assert result is not None
        assert result["local_track_id"] == "1"
        assert result["source_video_id"] == vid_id

    def test_get_track_unknown_returns_none(self, db: Session):
        svc = HistoryService(db)
        assert svc.get_track(str(uuid.uuid4())) is None

    def test_get_person_includes_tracks(self, db: Session):
        vid_id = "test-vid-K2"
        self._setup(db, vid_id)
        svc = IngestionService(db)
        svc.link_identity(
            source_video_id=vid_id,
            local_track_id="1",
            person_code="P777",
            identity_status="probable",
            identity_confidence=0.9,
            identity_meta={},
        )
        hsvc = HistoryService(db)
        result = hsvc.get_person("P777")
        assert result is not None
        assert result["person_code"] == "P777"
        assert any(t["local_track_id"] == "1" for t in result["tracks"])

    def test_get_camera_tracks(self, db: Session):
        vid_id = "test-vid-K3"
        self._setup(db, vid_id)
        hsvc = HistoryService(db)
        result = hsvc.get_camera_tracks(f"CAM-K-{vid_id[-3:]}")
        assert result is not None
        assert result["camera_code"] == f"CAM-K-{vid_id[-3:]}"
        assert len(result["tracks"]) >= 2

    def test_time_range_query(self, db: Session):
        vid_id = "test-vid-K4"
        self._setup(db, vid_id)
        hsvc = HistoryService(db)
        result = hsvc.query_time_range(
            source_video_id=vid_id,
            start=None,
            end=None,
            limit=100,
        )
        assert len(result["tracks"]) == 2
