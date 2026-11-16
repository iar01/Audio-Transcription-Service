"""Database engine/session management.

SQLite (default, local development) and PostgreSQL (production) are both
supported through ``DATABASE_URL``. SQLite is configured with WAL mode and a
busy timeout so the API process and worker processes can share one file.
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from config import get_settings
from database.models import Base

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_engine: Optional[Engine] = None
_session_factory: Optional[sessionmaker[Session]] = None


def _make_engine(url: str) -> Engine:
    if url.startswith("sqlite"):
        db_path = url.split("///", 1)[-1]
        if db_path and db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        engine = create_engine(
            url,
            connect_args={"check_same_thread": False, "timeout": 30},
            future=True,
        )

        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_conn, _record) -> None:  # type: ignore[no-untyped-def]
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA foreign_keys=ON")
            cur.execute("PRAGMA busy_timeout=30000")
            cur.execute("PRAGMA synchronous=NORMAL")
            cur.close()

        return engine

    return create_engine(url, pool_pre_ping=True, pool_size=5, max_overflow=10, future=True)


def get_engine() -> Engine:
    """Return the process-wide SQLAlchemy engine (created lazily)."""
    global _engine, _session_factory
    if _engine is None:
        with _lock:
            if _engine is None:
                settings = get_settings()
                _engine = _make_engine(settings.database_url)
                _session_factory = sessionmaker(bind=_engine, expire_on_commit=False, future=True)
    return _engine


def init_db() -> None:
    """Create tables if they don't exist (idempotent schema initialization)."""
    engine = get_engine()
    Base.metadata.create_all(engine)
    logger.info("Database schema initialized (%s)", engine.url.render_as_string(hide_password=True))


def reset_engine() -> None:
    """Dispose the engine (tests / after fork)."""
    global _engine, _session_factory
    with _lock:
        if _engine is not None:
            _engine.dispose()
        _engine = None
        _session_factory = None


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope: commit on success, rollback on error."""
    get_engine()
    assert _session_factory is not None
    session = _session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
