"""Identity decision fields, data source, and optional spatial position.

Revision ID: b7e1c0d2f3a4
Revises: a1b2c3d4e5f6
Create Date: 2026-10-15

Why these columns, and why they are deliberately loose:

tracks.identity_status
    The identity layer's verdict for this track (currently e.g. "probable",
    "uncertain", "new"). Kept as an open VARCHAR with NO check constraint,
    because the team handoff marks these values as provisional.
tracks.identity_confidence
    Optional 0..1 score for that verdict.
tracks.identity_meta (JSONB, never NULL)
    Bag for still-evolving evidence: candidate persons, appearance score,
    geometry verdict, model/version, decision reason. Lets the identity layer
    add fields without a migration each time.

    person_id stays nullable. "unresolved" is a valid, permanent state.

videos.source
    'cv' for real computer-vision output, 'simulated' for synthetic events.
    The project spec requires simulated results to be clearly separated.

observations.world_x / world_y / world_z / room_id
    Optional world/floor-plan position for a detection, filled only when
    the spatial layer can compute it.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "b7e1c0d2f3a4"
down_revision: Union[str, None] = "a1b2c3d4e5f6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── tracks: identity decision ─────────────────────────────────────────────
    op.add_column("tracks", sa.Column("identity_status", sa.String(30), nullable=True))
    op.add_column("tracks", sa.Column("identity_confidence", sa.Float, nullable=True))
    op.add_column(
        "tracks",
        sa.Column(
            "identity_meta",
            JSONB,
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
    )
    op.create_check_constraint(
        "ck_tracks_identity_confidence",
        "tracks",
        "identity_confidence IS NULL OR identity_confidence BETWEEN 0.0 AND 1.0",
    )

    # ── videos: real vs simulated ─────────────────────────────────────────────
    op.add_column(
        "videos",
        sa.Column("source", sa.String(20), nullable=False, server_default="cv"),
    )
    op.create_check_constraint(
        "ck_videos_source", "videos", "source IN ('cv', 'simulated')"
    )

    # ── observations: optional spatial position ──────────────────────────────
    op.add_column("observations", sa.Column("world_x", sa.Float, nullable=True))
    op.add_column("observations", sa.Column("world_y", sa.Float, nullable=True))
    op.add_column("observations", sa.Column("world_z", sa.Float, nullable=True))
    op.add_column(
        "observations",
        sa.Column("room_id", sa.Uuid(as_uuid=True), nullable=True),
    )
    op.create_foreign_key(
        "fk_observations_room",
        "observations",
        "rooms",
        ["room_id"],
        ["id"],
        ondelete="SET NULL",
    )
    # "Who was in room R between T1 and T2?"
    op.create_index("ix_observations_room_ts", "observations", ["room_id", "timestamp"])


def downgrade() -> None:
    op.drop_index("ix_observations_room_ts", table_name="observations")
    op.drop_constraint("fk_observations_room", "observations", type_="foreignkey")
    op.drop_column("observations", "room_id")
    op.drop_column("observations", "world_z")
    op.drop_column("observations", "world_y")
    op.drop_column("observations", "world_x")

    op.drop_constraint("ck_videos_source", "videos", type_="check")
    op.drop_column("videos", "source")

    op.drop_constraint("ck_tracks_identity_confidence", "tracks", type_="check")
    op.drop_column("tracks", "identity_meta")
    op.drop_column("tracks", "identity_confidence")
    op.drop_column("tracks", "identity_status")
