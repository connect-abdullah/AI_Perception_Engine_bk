from typing import Any

from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse


def ok(data: Any = None, message: str = "success", status_code: int = 200) -> JSONResponse:
    """Return a standard success envelope."""
    return JSONResponse(
        status_code=status_code,
        # jsonable_encoder: payloads contain datetimes/UUIDs, which plain JSON cannot encode.
        content=jsonable_encoder({"ok": True, "message": message, "data": data}),
    )


def fail(
    message: str = "error", status_code: int = 400, data: Any = None
) -> JSONResponse:
    """Return a standard error envelope."""
    return JSONResponse(
        status_code=status_code,
        content=jsonable_encoder({"ok": False, "message": message, "data": data}),
    )
