"""FastAPI application entry point.

Routes for this PoC:
    GET /          – application info
    GET /health    – database connectivity check

No CRUD routers are wired yet; the definition of done for this milestone is
schema creation and connectivity verification.
"""

from fastapi import Depends, FastAPI
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.config import settings
from app.core.logging import get_logger
from app.core.response import fail, ok
from app.db.session import get_db
from app.routes import api_router

logger = get_logger(__name__)

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="AI Perception Engine – video-first security PoC",
)

app.include_router(api_router)


@app.get("/")
def root():
    """Service identification."""
    return ok({"name": settings.app_name, "version": settings.app_version})


@app.get("/health")
def health(db: Session = Depends(get_db)):
    """Verify that the API can reach the database.

    Returns 200 + ok envelope when SELECT 1 succeeds; 503 + fail envelope
    when the database is unreachable.
    """
    try:
        db.execute(text("SELECT 1"))
        return ok({"database": "connected"})
    except Exception as exc:  # noqa: BLE001
        logger.error("Health check failed: %s", exc)
        return fail("Database connection failed", status_code=503)
