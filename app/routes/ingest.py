"""Ingestion routes – POST endpoints for perception, identity, activity, and events.

All routes:
  POST /api/v1/ingest/run       – perception run (observations.jsonl payload)
  POST /api/v1/ingest/identity  – link a local track to a global person
  POST /api/v1/ingest/activity  – add a state interval (Abubakar's output)
  POST /api/v1/ingest/event     – add a local event (Abubakar's output)
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.response import fail, ok
from app.db.session import get_db
from app.entities.ingest.schema import (
    ActivityIngestRequest,
    ActivityIngestResponse,
    EventIngestRequest,
    EventIngestResponse,
    IdentityLinkRequest,
    IdentityLinkResponse,
    RunIngestRequest,
    RunIngestResponse,
)
from app.entities.ingest.service import IngestionService

router = APIRouter(prefix="/ingest", tags=["ingest"])


@router.post("/run", response_model=None)
def ingest_run(body: RunIngestRequest, db: Session = Depends(get_db)):
    """Ingest one perception run.

    Idempotent: re-submitting the same source_video_id updates metadata and
    upserts observations without duplicating rows.
    """
    svc = IngestionService(db)
    try:
        result = svc.ingest_run(
            source_video_id=body.source_video_id,
            camera_code=body.camera_code,
            fps=body.fps,
            video_start_time=body.video_start_time,
            width=body.width,
            height=body.height,
            observations=[obs.model_dump() for obs in body.observations],
        )
        db.commit()
        return ok(RunIngestResponse(**result).model_dump())
    except ValueError as exc:
        db.rollback()
        return fail(str(exc), status_code=422)
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/identity", response_model=None)
def ingest_identity(body: IdentityLinkRequest, db: Session = Depends(get_db)):
    """Link a local track to a global person.

    Pass person_code=null to clear the identity assignment and mark the track
    as unresolved.
    """
    svc = IngestionService(db)
    try:
        result = svc.link_identity(
            source_video_id=body.source_video_id,
            local_track_id=body.local_track_id,
            person_code=body.person_code,
            identity_status=body.identity_status,
            identity_confidence=body.identity_confidence,
            identity_meta=body.identity_meta,
        )
        db.commit()
        return ok(IdentityLinkResponse(**result).model_dump())
    except ValueError as exc:
        db.rollback()
        return fail(str(exc), status_code=422)
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/activity", response_model=None)
def ingest_activity(body: ActivityIngestRequest, db: Session = Depends(get_db)):
    """Persist a state interval produced by Abubakar's activity layer."""
    svc = IngestionService(db)
    try:
        result = svc.ingest_activity(
            source_video_id=body.source_video_id,
            local_track_id=body.local_track_id,
            activity=body.activity,
            start_time=body.start_time,
            end_time=body.end_time,
            confidence=body.confidence,
        )
        db.commit()
        return ok(ActivityIngestResponse(**result).model_dump())
    except ValueError as exc:
        db.rollback()
        return fail(str(exc), status_code=422)
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/event", response_model=None)
def ingest_event(body: EventIngestRequest, db: Session = Depends(get_db)):
    """Persist a local event produced by Abubakar's event layer."""
    svc = IngestionService(db)
    try:
        result = svc.ingest_event(
            source_video_id=body.source_video_id,
            local_track_id=body.local_track_id,
            event_type=body.event_type,
            timestamp=body.timestamp,
            confidence=body.confidence,
            event_meta=body.event_meta,
        )
        db.commit()
        return ok(EventIngestResponse(**result).model_dump())
    except ValueError as exc:
        db.rollback()
        return fail(str(exc), status_code=422)
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(exc)) from exc
