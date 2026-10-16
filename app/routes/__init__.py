"""API router registry.

Mounts domain routers under /api/v1.
"""

from fastapi import APIRouter

from app.routes.history import router as history_router
from app.routes.ingest import router as ingest_router

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(ingest_router)
api_router.include_router(history_router)
