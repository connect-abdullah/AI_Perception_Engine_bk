"""Verify schema – confirms connectivity and that all eight tables exist.

Run after applying the schema with Alembic:
    poetry run python scripts/verify_schema.py

Exit code 0 = all tables present and database reachable.
Exit code 1 = failure (details printed to stderr).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv

load_dotenv()

from sqlalchemy import inspect, text  # noqa: E402

from app.db.session import engine  # noqa: E402

EXPECTED_TABLES = {
    "rooms",
    "cameras",
    "videos",
    "persons",
    "tracks",
    "observations",
    "state_intervals",
    "events",
}

errors: list[str] = []

# ── 1. Connectivity ───────────────────────────────────────────────────────────
try:
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    print("✓  Database connection OK")
except Exception as exc:
    print(f"✗  Cannot connect to database: {exc}", file=sys.stderr)
    sys.exit(1)

# ── 2. Table presence ─────────────────────────────────────────────────────────
insp = inspect(engine)
existing = set(insp.get_table_names(schema="public"))

for table in sorted(EXPECTED_TABLES):
    if table in existing:
        print(f"✓  Table exists: {table}")
    else:
        msg = f"✗  Missing table: {table}"
        print(msg, file=sys.stderr)
        errors.append(msg)

# ── 3. Result ─────────────────────────────────────────────────────────────────
if errors:
    print(f"\n{len(errors)} table(s) missing. Run: poetry run alembic upgrade head",
          file=sys.stderr)
    sys.exit(1)

print("\nAll eight tables present. Schema verification passed.")
