"""Central configuration for Volga, read from environment variables.

Every tunable (Redis URL, database URL, storage, worker concurrency, chunking,
retries) lives here so nothing is hardcoded elsewhere. Values can be supplied
through real environment variables or a ``.env`` file in the project root
(real environment variables take precedence).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent


def _load_dotenv(path: Path) -> None:
    """Minimal ``.env`` loader (KEY=VALUE lines, # comments). Does not override real env vars."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value if value else default


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    try:
        return int(value)
    except ValueError as err:
        raise ValueError(f"Environment variable {name} must be an integer, got {value!r}") from err


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    try:
        return float(value)
    except ValueError as err:
        raise ValueError(f"Environment variable {name} must be a number, got {value!r}") from err


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name, "").strip().lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "on"}


def _default_pool() -> str:
    # Celery's prefork pool is not supported on Windows; 'solo' runs one task at a time per process.
    return "solo" if sys.platform.startswith("win") else "prefork"


@dataclass(frozen=True)
class Settings:
    """Immutable application settings."""

    # Infrastructure
    redis_url: str
    database_url: str

    # Celery workers
    celery_concurrency: int
    celery_pool: str
    celery_max_retries: int
    celery_retry_backoff_base: int
    celery_retry_backoff_max: int
    celery_visibility_timeout: int
    celery_task_always_eager: bool
    celery_queue: str

    # Chunking
    audio_chunk_duration: int
    audio_chunk_overlap: int
    chunking_threshold: int
    chunking_enabled: bool

    # Recovery
    chunk_stale_timeout: int
    queued_stale_timeout: int

    # Storage
    storage_backend: str
    storage_path: Path
    retention_days: int

    # Uploads
    max_upload_size_mb: int

    # Engine
    transcription_engine: str
    whisper_model: str
    whisper_device: str
    whisper_compute_type: str
    whisper_cpu_threads: int
    whisper_vad_filter: bool

    # API / SSE
    sse_poll_interval: float
    sse_heartbeat_interval: float
    enable_sync_transcribe: bool
    log_level: str

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024


def _resolve_storage_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else (BASE_DIR / path)


def load_settings() -> Settings:
    """Build Settings from the environment (and ``.env`` if present)."""
    _load_dotenv(BASE_DIR / ".env")

    storage_path = _resolve_storage_path(_env_str("STORAGE_PATH", "storage"))
    default_db = f"sqlite:///{(storage_path / 'volga.db').as_posix()}"

    chunk_duration = _env_int("AUDIO_CHUNK_DURATION", 300)
    chunk_overlap = _env_int("AUDIO_CHUNK_OVERLAP", 3)
    if chunk_duration <= 0:
        raise ValueError("AUDIO_CHUNK_DURATION must be positive")
    if chunk_overlap < 0 or chunk_overlap * 2 >= chunk_duration:
        raise ValueError("AUDIO_CHUNK_OVERLAP must be >= 0 and less than half of AUDIO_CHUNK_DURATION")

    return Settings(
        redis_url=_env_str("REDIS_URL", "redis://localhost:6379/0"),
        database_url=_env_str("DATABASE_URL", default_db),
        celery_concurrency=max(1, _env_int("CELERY_CONCURRENCY", 1)),
        celery_pool=_env_str("CELERY_POOL", _default_pool()),
        celery_max_retries=max(0, _env_int("CELERY_MAX_RETRIES", 3)),
        celery_retry_backoff_base=max(1, _env_int("CELERY_RETRY_BACKOFF_BASE", 5)),
        celery_retry_backoff_max=max(1, _env_int("CELERY_RETRY_BACKOFF_MAX", 300)),
        celery_visibility_timeout=max(60, _env_int("CELERY_VISIBILITY_TIMEOUT", 3600)),
        celery_task_always_eager=_env_bool("CELERY_TASK_ALWAYS_EAGER", False),
        celery_queue=_env_str("CELERY_QUEUE", "transcription"),
        audio_chunk_duration=chunk_duration,
        audio_chunk_overlap=chunk_overlap,
        chunking_threshold=max(0, _env_int("CHUNKING_THRESHOLD", 600)),
        chunking_enabled=_env_bool("CHUNKING_ENABLED", True),
        chunk_stale_timeout=max(5, _env_int("CHUNK_STALE_TIMEOUT", 1800)),
        queued_stale_timeout=max(5, _env_int("QUEUED_STALE_TIMEOUT", 300)),
        storage_backend=_env_str("STORAGE_BACKEND", "local").lower(),
        storage_path=storage_path,
        retention_days=max(0, _env_int("RETENTION_DAYS", 30)),
        max_upload_size_mb=max(1, _env_int("MAX_UPLOAD_SIZE_MB", 500)),
        transcription_engine=_env_str("TRANSCRIPTION_ENGINE", "auto").lower(),
        whisper_model=_env_str("WHISPER_MODEL", "base"),
        whisper_device=_env_str("WHISPER_DEVICE", "cpu"),
        whisper_compute_type=_env_str("WHISPER_COMPUTE_TYPE", "int8"),
        whisper_cpu_threads=max(0, _env_int("WHISPER_CPU_THREADS", 0)),
        whisper_vad_filter=_env_bool("WHISPER_VAD_FILTER", True),
        sse_poll_interval=max(0.1, _env_float("SSE_POLL_INTERVAL", 1.0)),
        sse_heartbeat_interval=max(1.0, _env_float("SSE_HEARTBEAT_INTERVAL", 15.0)),
        enable_sync_transcribe=_env_bool("ENABLE_SYNC_TRANSCRIBE", True),
        log_level=_env_str("LOG_LEVEL", "INFO").upper(),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return process-wide cached settings."""
    return load_settings()


def reset_settings_cache() -> Optional[Settings]:
    """Clear the cached settings (used by tests after changing env vars)."""
    get_settings.cache_clear()
    return None
