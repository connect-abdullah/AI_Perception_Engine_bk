"""API router registry.

For this PoC the only surface is /health (mounted directly in main.py).
Domain routers are added here when they exist:

    from app.routes import rooms, cameras, videos
    api_router = APIRouter(prefix="/api/v1")
    api_router.include_router(rooms.router)
    ...
"""
