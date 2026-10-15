from collections.abc import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from app.config import settings

# pool_pre_ping sends a lightweight SELECT 1 before returning a pooled
# connection, so stale connections from an idle Supabase session are recycled.
engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
    # Supabase's transaction pooler (port 6543) cannot reuse server-side
    # prepared statements; disabling them avoids "prepared statement already
    # exists" errors. Harmless on a direct connection.
    connect_args={"prepare_threshold": None},
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency – yield a scoped DB session, close it on exit."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
