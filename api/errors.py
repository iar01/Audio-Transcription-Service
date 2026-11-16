"""Consistent API error responses.

Every error from ``/api/v1`` has this shape::

    {"error": {"code": "unsupported_format", "message": "..."}, "detail": "..."}

``detail`` duplicates the message so existing clients that read FastAPI's
``detail`` field keep working. Internal paths and stack traces are never included.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import Request
from fastapi.responses import JSONResponse


class ApiError(Exception):
    def __init__(self, status_code: int, code: str, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.extra = extra or {}


def error_body(code: str, message: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    body: Dict[str, Any] = {"error": {"code": code, "message": message}, "detail": message}
    body.update(extra or {})
    return body


async def api_error_handler(_request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, ApiError)
    return JSONResponse(status_code=exc.status_code, content=error_body(exc.code, exc.message, exc.extra))


def not_found(job_id: str) -> ApiError:
    return ApiError(404, "job_not_found", f"Transcription job '{job_id}' was not found.")
