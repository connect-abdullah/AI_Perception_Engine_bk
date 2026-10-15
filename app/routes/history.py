"""History routes – GET endpoints for track, person, camera, and time-range queries.

All routes:
  GET /api/v1/history/tracks/{track_id}           – one track by UUID
  GET /api/v1/history/persons/{person_code}        – person + linked tracks
  GET /api/v1/history/cameras/{camera_code}/tracks – all tracks for a camera
  GET /api/v1/history/time-range                   – tracks in a time window
"""

from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.core.response import fail, ok
from app.db.session import get_db
from app.entities.ingest.service import HistoryService

router = APIRouter(prefix="/history", tags=["history"])


@router.get("/tracks/{track_id}", response_model=None)
def get_track(track_id: str, db: Session = Depends(get_db)):
    """Retrieve one track by its internal UUID."""
    svc = HistoryService(db)
    result = svc.get_track(track_id)
    if result is None:
        return fail(f"Track {track_id!r} not found", status_code=404)
    return ok(result)


@router.get("/persons/{person_code}", response_model=None)
def get_person(person_code: str, db: Session = Depends(get_db)):
    """Retrieve a global person and all tracks assigned to it."""
    svc = HistoryService(db)
    result = svc.get_person(person_code)
    if result is None:
        return fail(f"Person {person_code!r} not found", status_code=404)
    return ok(result)


@router.get("/cameras/{camera_code}/tracks", response_model=None)
def get_camera_tracks(
    camera_code: str,
    limit: int = Query(default=100, ge=1, le=1000),
    db: Session = Depends(get_db),
):
    """All tracks observed on a camera (across all videos)."""
    svc = HistoryService(db)
    result = svc.get_camera_tracks(camera_code, limit=limit)
    if result is None:
        return fail(f"Camera {camera_code!r} not found", status_code=404)
    return ok(result)


@router.get("/time-range", response_model=None)
def query_time_range(
    camera_code: Optional[str] = Query(default=None),
    source_video_id: Optional[str] = Query(default=None),
    start: Optional[str] = Query(default=None, description="ISO-8601 datetime"),
    end: Optional[str] = Query(default=None, description="ISO-8601 datetime"),
    limit: int = Query(default=100, ge=1, le=1000),
    db: Session = Depends(get_db),
):
    """Tracks whose start_time falls within the given window.

    Filter by camera_code or source_video_id (both optional).
    """
    from datetime import datetime

    def _parse(s: Optional[str]) -> Optional[datetime]:
        if s is None:
            return None
        try:
            return datetime.fromisoformat(s)
        except ValueError:
            return None

    svc = HistoryService(db)
    result = svc.query_time_range(
        camera_code=camera_code,
        source_video_id=source_video_id,
        start=_parse(start),
        end=_parse(end),
        limit=limit,
    )
    return ok(result)
