"""Alembic environment configuration.

Uses DIRECT_URL (port 5432, session mode) for migrations because the Supabase
transaction-mode pooler does not support DDL statements.  Falls back to
DATABASE_URL when DIRECT_URL is not set.
"""

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

# Import Base + all models so metadata is fully populated for autogenerate.
from app.db.base import Base  # noqa: F401 (side-effect: registers all models)
from app.config import settings

# ── Alembic config object ─────────────────────────────────────────────────────
config = context.config

# Use DIRECT_URL for DDL; fall back to DATABASE_URL.
migration_url = settings.direct_url or settings.database_url
config.set_main_option("sqlalchemy.url", migration_url)

# Logging setup from alembic.ini.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


# ── Offline mode (no live connection required) ────────────────────────────────
def run_migrations_offline() -> None:
    """Emit SQL to stdout instead of connecting to the database."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


# ── Online mode ───────────────────────────────────────────────────────────────
def run_migrations_online() -> None:
    """Run migrations against a live database connection."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,  # NullPool is correct for migration scripts.
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
