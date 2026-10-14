from typing import Any

from fastapi.responses import JSONResponse


def ok(data: Any = None, message: str = "success", status_code: int = 200) -> JSONResponse:
    """Return a standard success envelope."""
    return JSONResponse(
        status_code=status_code,
        content={"ok": True, "message": message, "data": data},
    )


def fail(
    message: str = "error", status_code: int = 400, data: Any = None
) -> JSONResponse:
    """Return a standard error envelope."""
    return JSONResponse(
        status_code=status_code,
        content={"ok": False, "message": message, "data": data},
    )
