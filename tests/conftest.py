"""Shared fixtures for the asynchronous (database / Celery) test suites.

Existing pipeline tests don't use these fixtures and are unaffected.
"""

from __future__ import annotations

import math
import shutil
import struct
import wave
from pathlib import Path
from typing import Callable, Dict, Iterator

import pytest

HAS_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
requires_ffmpeg = pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg/ffprobe not installed")


def make_wav(path: Path, seconds: float, rate: int = 16000) -> Path:
    """Write a mono 16-bit sine-tone WAV of the given length."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = int(seconds * rate)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        chunk = 16000
        for start in range(0, frames, chunk):
            n = min(chunk, frames - start)
            w.writeframes(b"".join(
                struct.pack("<h", int(2000 * math.sin(2 * math.pi * 440 * (start + i) / rate))) for i in range(n)
            ))
    return path


def configure_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, overrides: Dict[str, str]) -> None:
    """Point every Volga component at an isolated temp environment and reset cached singletons."""
    env = {
        "STORAGE_PATH": str(tmp_path / "storage"),
        "DATABASE_URL": "",
        "CELERY_TASK_ALWAYS_EAGER": "true",
        "TRANSCRIPTION_ENGINE": "mock",
        "AUDIO_CHUNK_DURATION": "10",
        "AUDIO_CHUNK_OVERLAP": "2",
        "CHUNKING_THRESHOLD": "15",
        "CELERY_MAX_RETRIES": "3",
        "CELERY_RETRY_BACKOFF_BASE": "1",
        "CELERY_RETRY_BACKOFF_MAX": "1",
        "SSE_POLL_INTERVAL": "0.1",
        "SSE_HEARTBEAT_INTERVAL": "5",
        "MAX_UPLOAD_SIZE_MB": "50",
        "RETENTION_DAYS": "30",
    }
    env.update(overrides)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    reset_singletons()


def reset_singletons() -> None:
    from config import get_settings, reset_settings_cache
    from database.session import init_db, reset_engine
    from filestore import reset_storage
    from services import transcription_service
    from workers.celery_app import apply_settings, celery_app

    reset_settings_cache()
    reset_engine()
    reset_storage()
    transcription_service.reset_engine()
    transcription_service.engine_factory = None
    apply_settings(celery_app, get_settings())
    init_db()


@pytest.fixture
def volga_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Callable[..., None]]:
    """Eager-mode environment (tasks run in-process). Call it again with overrides to reconfigure."""

    def _configure(**overrides: str) -> None:
        configure_env(monkeypatch, tmp_path, overrides)

    _configure()
    yield _configure
    monkeypatch.undo()
    reset_singletons()


@pytest.fixture
def client(volga_env: Callable[..., None]):  # type: ignore[no-untyped-def]
    from fastapi.testclient import TestClient

    import app as app_module

    with TestClient(app_module.app) as test_client:
        yield test_client
