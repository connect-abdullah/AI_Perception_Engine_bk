"""Fixtures for Postgres integration tests.

These tests require a running Postgres database.  The default URL targets the
local Docker Compose instance:

    docker compose up -d
    TEST_DATABASE_URL=postgresql+psycopg://perception:perception@localhost:5432/perception?sslmode=disable

A test-specific schema (perception_test) is created, migrated, and dropped for
each test session, leaving the development database untouched.

Skip all tests in this module if the database is unreachable:

    pytest tests/integration/ -v
    pytest tests/integration/ -v --timeout=30

The tests are NOT run as part of the offline test suite (they require Postgres).
They have the marker `integration` and are excluded from the default run unless
the DATABASE_URL is set.
"""

from __future__ import annotations

import os
import subprocess
import sys
import uuid

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_URL = (
    "postgresql+psycopg://perception:perception@localhost:5432/perception"
    "?sslmode=disable"
)
TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL", _DEFAULT_URL)


def _check_db_reachable(url: str) -> bool:
    try:
        eng = create_engine(url, pool_pre_ping=True, connect_args={"prepare_threshold": None})
        with eng.connect() as conn:
            conn.execute(text("SELECT 1"))
        eng.dispose()
        return True
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Session-scoped engine: runs migrations once, tears down at the end.
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def db_url():
    """Return the test database URL or skip if Postgres is unreachable."""
    if not _check_db_reachable(TEST_DATABASE_URL):
        pytest.skip(
            f"Postgres not reachable at {TEST_DATABASE_URL}. "
            "Start it with: docker compose up -d"
        )
    return TEST_DATABASE_URL


@pytest.fixture(scope="session")
def migrated_engine(db_url):
    """Create an engine after running Alembic migrations from scratch.

    Uses a separate test schema suffix to avoid colliding with dev data.
    Drops and recreates all tables between test sessions.
    """
    engine = create_engine(
        db_url,
        pool_pre_ping=True,
        connect_args={"prepare_threshold": None},
    )

    # Drop ALL tables (including alembic_version) so migrations start from scratch.
    from app.db.base import Base
    with engine.begin() as conn:
        Base.metadata.drop_all(conn)
        # alembic_version is not in Base.metadata; drop it explicitly so
        # Alembic does not think it is already at head.
        conn.execute(text("DROP TABLE IF EXISTS alembic_version"))

    # Run Alembic upgrade head via the venv Python (has psycopg + alembic).
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    alembic_bin = os.path.join(os.path.dirname(sys.executable), "alembic")
    if not os.path.exists(alembic_bin):
        # Fall back to the module form if the bin-wrapper is absent.
        cmd = [sys.executable, "-c",
               "from alembic.config import main; main(argv=['upgrade','head'])"]
    else:
        cmd = [alembic_bin, "upgrade", "head"]
    result = subprocess.run(
        cmd,
        cwd=project_root,
        env={**os.environ, "DIRECT_URL": db_url, "DATABASE_URL": db_url},
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.fail(
            f"Alembic upgrade failed:\n{result.stdout}\n{result.stderr}"
        )

    yield engine

    engine.dispose()


@pytest.fixture(scope="function")
def db(migrated_engine) -> Session:
    """Yield a per-test session wrapped in a savepoint; roll back after the test.

    This keeps tests isolated without recreating the schema for every test.
    """
    SessionLocal = sessionmaker(bind=migrated_engine, autocommit=False, autoflush=False)
    session = SessionLocal()
    session.begin_nested()  # SAVEPOINT
    try:
        yield session
    finally:
        session.rollback()
        session.close()
