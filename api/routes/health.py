"""Dependency health check for the asynchronous stack (database + Redis broker)."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from sqlalchemy import text

from config import get_settings
from database.session import get_engine

router = APIRouter(prefix="/api/v1", tags=["health"])


@router.get("/health")
def health() -> JSONResponse:
    checks: Dict[str, Any] = {}
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception:
        checks["database"] = "unavailable"

    try:
        import redis

        redis.Redis.from_url(get_settings().redis_url, socket_connect_timeout=2, socket_timeout=2).ping()
        checks["redis"] = "ok"
    except Exception:
        checks["redis"] = "unavailable"

    healthy = all(v == "ok" for v in checks.values())
    return JSONResponse(status_code=200 if healthy else 503,
                        content={"status": "ok" if healthy else "degraded", "checks": checks})
