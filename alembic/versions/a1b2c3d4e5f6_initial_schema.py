"""Initial schema – all eight perception tables.

Revision ID: a1b2c3d4e5f6
Revises: –
Create Date: 2026-10-14

This revision is hand-written (autogenerate requires a live connection).
It is the authoritative source of truth for the initial schema; it mirrors
sql/schema.sql exactly.

After you fill in .env with your Supabase DIRECT_URL run:

    poetry run alembic upgrade head
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── Enable pgcrypto (gen_random_uuid) ─────────────────────────────────────
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")

    # ── rooms ─────────────────────────────────────────────────────────────────
    op.create_table(
        "rooms",
        sa.Column(
            "id",
            sa.Uuid(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("name",  sa.String(120), nullable=False),
        sa.Column("floor", sa.Integer,     nullable=True),
    )

    # ── cameras ───────────────────────────────────────────────────────────────
    op.create_table(
        "cameras",
        sa.Column(
            "id",
            sa.Uuid(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("camera_code", sa.String(50), nullable=False),
        sa.Column("room_id",     sa.Uuid(as_uuid=True), nullable=True),
        sa.UniqueConstraint("camera_code", name="uq_cameras_code"),
        sa.ForeignKeyConstraint(
            ["room_id"], ["rooms.id"],
            name="fk_cameras_room", ondelete="SET NULL",
        ),
    )
    op.create_index("ix_cameras_room_id", "cameras", ["room_id"])

    # ── videos ────────────────────────────────────────────────────────────────
    op.create_table(
        "videos",
        sa.Column(
            "id",
            sa.Uuid(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("camera_id",        sa.Uuid(as_uuid=True),      nullable=False),
        sa.Column("file_path",        sa.Text,                     nullable=False),
        sa.Column("video_start_time", sa.DateTime(timezone=True),  nullable=False),
        sa.Column("fps",              sa.Float,                    nullable=True),
        sa.Column("width",            sa.Integer,                  nullable=True),
        sa.Column("height",           sa.Integer,                  nullable=True),
        sa.Column("duration_seconds", sa.Float,                    nullable=True),
        sa.Column(
            "status",
            sa.String(20),
            nullable=False,
            server_default="uploaded",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.ForeignKeyConstraint(
            ["camera_id"], ["cameras.id"],
            name="fk_videos_camera", ondelete="RESTRICT",
        ),
        sa.CheckConstraint(
            "status IN ('uploaded','processing','completed','failed')",
            name="ck_videos_status",
        ),
        sa.CheckConstraint("fps IS NULL OR fps > 0",                  name="ck_videos_fps"),
        sa.CheckConstraint("width IS NULL OR width > 0",              name="ck_videos_width"),
        sa.CheckConstraint("height IS NULL OR height > 0",            name="ck_videos_height"),
        sa.CheckConstraint(
            "duration_seconds IS NULL OR duration_seconds >= 0",
            name="ck_videos_duration",
        ),
    )
    op.create_index("ix_videos_camera_id",  "videos", ["camera_id"])
    op.create_index("ix_videos_start_time", "videos", ["video_start_time"])

    # ── persons ───────────────────────────────────────────────────────────────
    op.create_table(
        "persons",
        sa.Column(
            "id",
            sa.Uuid(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("person_code", sa.String(20),  nullable=False),
        sa.Column("known_name",  sa.String(120), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint("person_code", name="uq_persons_code"),
    )

    # ── tracks ────────────────────────────────────────────────────────────────
    op.create_table(
        "tracks",
        sa.Column(
            "id",
            sa.Uuid(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("video_id",       sa.Uuid(as_uuid=True),     nullable=False),
        sa.Column("local_track_id", sa.String(80),              nullable=False),
        sa.Column("person_id",      sa.Uuid(as_uuid=True),     nullable=True),
        sa.Column("start_time",     sa.DateTime(timezone=True), nullable=False),
        sa.Column("end_time",       sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("video_id", "local_track_id", name="uq_tracks_video_local"),
        sa.ForeignKeyConstraint(
            ["video_id"], ["videos.id"],
            name="fk_tracks_video", ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["person_id"], ["persons.id"],
            name="fk_tracks_person", ondelete="SET NULL",
        ),
        sa.CheckConstraint(
            "end_time IS NULL OR end_time >= start_time",
            name="ck_tracks_end_after_start",
        ),
    )
    op.create_index("ix_tracks_video_id",     "tracks", ["video_id"])
    op.create_index("ix_tracks_person_start", "tracks", ["person_id", "start_time"])

    # ── observations ──────────────────────────────────────────────────────────
    op.create_table(
        "observations",
        sa.Column("id",       sa.BigInteger, sa.Identity(always=True), primary_key=True),
        sa.Column("track_id", sa.Uuid(as_uuid=True),     nullable=False),
        sa.Column("frame_number",         sa.Integer,                  nullable=False),
        sa.Column("timestamp",            sa.DateTime(timezone=True),  nullable=False),
        sa.Column("bbox_x",               sa.Integer,                  nullable=False),
        sa.Column("bbox_y",               sa.Integer,                  nullable=False),
        sa.Column("bbox_width",           sa.Integer,                  nullable=False),
        sa.Column("bbox_height",          sa.Integer,                  nullable=False),
        sa.Column("detection_confidence", sa.Float,                    nullable=False),
        sa.Column("activity",             sa.String(30),               nullable=True),
        sa.Column("activity_confidence",  sa.Float,                    nullable=True),
        sa.UniqueConstraint(
            "track_id", "frame_number", name="uq_observations_track_frame"
        ),
        sa.ForeignKeyConstraint(
            ["track_id"], ["tracks.id"],
            name="fk_observations_track", ondelete="CASCADE",
        ),
        sa.CheckConstraint("frame_number >= 0",    name="ck_obs_frame_non_negative"),
        sa.CheckConstraint("bbox_width  > 0",      name="ck_obs_bbox_width"),
        sa.CheckConstraint("bbox_height > 0",      name="ck_obs_bbox_height"),
        sa.CheckConstraint(
            "detection_confidence BETWEEN 0.0 AND 1.0", name="ck_obs_det_confidence"
        ),
        sa.CheckConstraint(
            "activity_confidence IS NULL OR activity_confidence BETWEEN 0.0 AND 1.0",
            name="ck_obs_act_confidence",
        ),
    )
    op.create_index("ix_observations_track_ts", "observations", ["track_id", "timestamp"])

    # ── state_intervals ───────────────────────────────────────────────────────
    op.create_table(
        "state_intervals",
        sa.Column(
            "id",
            sa.Uuid(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("track_id",   sa.Uuid(as_uuid=True),     nullable=False),
        sa.Column("activity",   sa.String(30),              nullable=False),
        sa.Column("start_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("end_time",   sa.DateTime(timezone=True), nullable=True),
        sa.Column("confidence", sa.Float,                   nullable=True),
        sa.ForeignKeyConstraint(
            ["track_id"], ["tracks.id"],
            name="fk_si_track", ondelete="CASCADE",
        ),
        sa.CheckConstraint(
            "end_time IS NULL OR end_time >= start_time",
            name="ck_si_end_after_start",
        ),
        sa.CheckConstraint(
            "confidence IS NULL OR confidence BETWEEN 0.0 AND 1.0",
            name="ck_si_confidence",
        ),
    )
    op.create_index("ix_si_track_id",    "state_intervals", ["track_id"])
    op.create_index("ix_si_track_start", "state_intervals", ["track_id", "start_time"])

    # ── events ────────────────────────────────────────────────────────────────
    op.create_table(
        "events",
        sa.Column(
            "id",
            sa.Uuid(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("track_id",   sa.Uuid(as_uuid=True),     nullable=True),
        sa.Column("event_type", sa.String(50),              nullable=False),
        sa.Column("timestamp",  sa.DateTime(timezone=True), nullable=False),
        sa.Column("room_id",    sa.Uuid(as_uuid=True),     nullable=True),
        sa.Column("confidence", sa.Float,                   nullable=True),
        sa.Column(
            "metadata",
            JSONB,
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.ForeignKeyConstraint(
            ["track_id"], ["tracks.id"],
            name="fk_events_track", ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["room_id"], ["rooms.id"],
            name="fk_events_room", ondelete="SET NULL",
        ),
        sa.CheckConstraint(
            "confidence IS NULL OR confidence BETWEEN 0.0 AND 1.0",
            name="ck_events_confidence",
        ),
    )
    op.create_index("ix_events_room_ts",  "events", ["room_id",    "timestamp"])
    op.create_index("ix_events_track_id", "events", ["track_id"])
    op.create_index("ix_events_type_ts",  "events", ["event_type", "timestamp"])


def downgrade() -> None:
    # Drop in reverse FK dependency order.
    op.drop_table("events")
    op.drop_table("state_intervals")
    op.drop_table("observations")
    op.drop_table("tracks")
    op.drop_table("persons")
    op.drop_table("videos")
    op.drop_table("cameras")
    op.drop_table("rooms")
