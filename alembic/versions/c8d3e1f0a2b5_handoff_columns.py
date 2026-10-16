"""Add videos.source_video_id and observations.crop_path.

Revision ID: c8d3e1f0a2b5
Revises: b7e1c0d2f3a4
Create Date: 2026-10-15

Why these columns:

videos.source_video_id
    The perception producer assigns a stable string identifier to each
    processed video run (by default "sha256:<hex>").  Persisting it lets
    the ingestion layer look up or upsert a Video row by the same ID that
    appears in manifest.json and observations.jsonl, without the caller
    needing to know the internal UUID up front.

    Column is nullable so that rows registered before ingestion (e.g. camera
    enrollment) remain valid.  A unique constraint prevents two perception runs
    from claiming the same source_video_id.

observations.crop_path
    The handoff contract already includes crop_path in each observation line.
    Persisting it keeps the database and the file handoff reconcilable: a
    consumer can find the crop PNG for any DB observation without re-reading
    observations.jsonl.

    Nullable because observations imported without a matching crop (or created
    by the activity layer alone) do not have one.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "c8d3e1f0a2b5"
down_revision: Union[str, None] = "b7e1c0d2f3a4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── videos: stable handoff string identifier ─────────────────────────────
    op.add_column(
        "videos",
        sa.Column("source_video_id", sa.Text, nullable=True),
    )
    op.create_unique_constraint(
        "uq_videos_source_video_id", "videos", ["source_video_id"]
    )
    # Support "find the video row for this run by its handoff ID".
    op.create_index("ix_videos_source_video_id", "videos", ["source_video_id"])

    # ── observations: crop path from the handoff contract ────────────────────
    op.add_column(
        "observations",
        sa.Column("crop_path", sa.Text, nullable=True),
    )


def downgrade() -> None:
    op.drop_column("observations", "crop_path")

    op.drop_index("ix_videos_source_video_id", table_name="videos")
    op.drop_constraint("uq_videos_source_video_id", "videos", type_="unique")
    op.drop_column("videos", "source_video_id")
