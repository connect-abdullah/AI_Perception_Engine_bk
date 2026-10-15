"""Metadata-only tests – no database connection required.

These tests verify that:
  1. All eight SQLAlchemy table names are registered on Base.metadata.
  2. Key unique constraints and check constraints are present on each table.

Run with:
    poetry run pytest tests/ -v
"""

import pytest
import sqlalchemy as sa

# Side-effect import: registers all eight models on Base.metadata.
from app.db.base import Base


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def metadata() -> sa.MetaData:
    return Base.metadata


@pytest.fixture(scope="module")
def tables(metadata: sa.MetaData) -> dict[str, sa.Table]:
    return metadata.tables


# ── Table presence ────────────────────────────────────────────────────────────

EXPECTED_TABLES = [
    "rooms",
    "cameras",
    "videos",
    "persons",
    "tracks",
    "observations",
    "state_intervals",
    "events",
]


@pytest.mark.parametrize("name", EXPECTED_TABLES)
def test_table_registered(tables, name):
    assert name in tables, f"Table '{name}' not found in Base.metadata"


# ── Column presence (spot checks) ─────────────────────────────────────────────

def test_observations_has_identity_pk(tables):
    """observations.id must use BIGINT GENERATED ALWAYS AS IDENTITY."""
    col = tables["observations"].c["id"]
    assert isinstance(col.type, sa.BigInteger)
    assert col.identity is not None, "observations.id should have Identity()"


def test_events_metadata_jsonb(tables):
    from sqlalchemy.dialects.postgresql import JSONB
    col = tables["events"].c["metadata"]
    assert isinstance(col.type, JSONB), "events.metadata should be JSONB"
    assert not col.nullable, "events.metadata must be NOT NULL"


def test_tracks_person_id_nullable(tables):
    col = tables["tracks"].c["person_id"]
    assert col.nullable, "tracks.person_id must be nullable (unresolved ReID)"


# ── Unique constraints ────────────────────────────────────────────────────────

def _unique_constraint_columns(table: sa.Table) -> list[frozenset[str]]:
    return [
        frozenset(c.name for c in uc.columns)
        for uc in table.constraints
        if isinstance(uc, sa.UniqueConstraint)
    ]


def test_cameras_unique_camera_code(tables):
    uniques = _unique_constraint_columns(tables["cameras"])
    assert frozenset({"camera_code"}) in uniques


def test_persons_unique_person_code(tables):
    uniques = _unique_constraint_columns(tables["persons"])
    assert frozenset({"person_code"}) in uniques


def test_tracks_unique_video_local(tables):
    uniques = _unique_constraint_columns(tables["tracks"])
    assert frozenset({"video_id", "local_track_id"}) in uniques


def test_observations_unique_track_frame(tables):
    uniques = _unique_constraint_columns(tables["observations"])
    assert frozenset({"track_id", "frame_number"}) in uniques


# ── Check constraints ─────────────────────────────────────────────────────────

def _check_names(table: sa.Table) -> set[str]:
    return {
        c.name
        for c in table.constraints
        if isinstance(c, sa.CheckConstraint) and c.name
    }


def test_videos_status_check(tables):
    assert "ck_videos_status" in _check_names(tables["videos"])


def test_observations_confidence_checks(tables):
    names = _check_names(tables["observations"])
    assert "ck_obs_det_confidence" in names
    assert "ck_obs_act_confidence" in names
    assert "ck_obs_bbox_width" in names
    assert "ck_obs_bbox_height" in names
    assert "ck_obs_frame_non_negative" in names


def test_tracks_end_after_start(tables):
    assert "ck_tracks_end_after_start" in _check_names(tables["tracks"])


def test_state_intervals_checks(tables):
    names = _check_names(tables["state_intervals"])
    assert "ck_si_end_after_start" in names
    assert "ck_si_confidence" in names


def test_events_confidence_check(tables):
    assert "ck_events_confidence" in _check_names(tables["events"])


# ── Foreign key delete rules ──────────────────────────────────────────────────

def _fk_ondelete(table: sa.Table, col_name: str) -> str | None:
    for fk in table.foreign_keys:
        if fk.parent.name == col_name:
            return fk.ondelete
    return None


def test_videos_camera_id_restrict(tables):
    assert _fk_ondelete(tables["videos"], "camera_id") == "RESTRICT"


def test_tracks_video_id_cascade(tables):
    assert _fk_ondelete(tables["tracks"], "video_id") == "CASCADE"


def test_tracks_person_id_set_null(tables):
    assert _fk_ondelete(tables["tracks"], "person_id") == "SET NULL"


def test_observations_track_id_cascade(tables):
    assert _fk_ondelete(tables["observations"], "track_id") == "CASCADE"


def test_events_track_id_set_null(tables):
    assert _fk_ondelete(tables["events"], "track_id") == "SET NULL"


# ── Migration b7e1c0d2f3a4: identity decision, source, spatial ───────────────

def test_tracks_identity_fields_are_optional(tables):
    t = tables["tracks"]
    assert t.c["identity_status"].nullable, "unresolved tracks have no status yet"
    assert t.c["identity_confidence"].nullable
    assert not t.c["identity_meta"].nullable, "identity_meta defaults to {}"


def test_tracks_identity_confidence_check(tables):
    assert "ck_tracks_identity_confidence" in _check_names(tables["tracks"])


def test_videos_source_column(tables):
    assert not tables["videos"].c["source"].nullable
    assert "ck_videos_source" in _check_names(tables["videos"])


def test_observations_spatial_fields_optional(tables):
    t = tables["observations"]
    for col in ("world_x", "world_y", "world_z", "room_id"):
        assert t.c[col].nullable, f"observations.{col} must be optional"
    assert _fk_ondelete(t, "room_id") == "SET NULL"
