"""Storage backend interface and key layout.

Keys are POSIX-style relative strings (``audio/<user>/<job>/original.wav``)
so a future S3/MinIO backend can use them unchanged as object keys.
"""

from __future__ import annotations

import re
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Protocol, runtime_checkable

_SAFE_SEGMENT = re.compile(r"^[A-Za-z0-9_.-]+$")


class StorageError(Exception):
    """Raised for storage backend failures (invalid keys, I/O errors)."""


class ObjectNotFoundError(StorageError):
    """Raised when a requested object does not exist (a permanent error for tasks)."""


def _safe(segment: str) -> str:
    if not segment or segment in {".", ".."} or not _SAFE_SEGMENT.match(segment):
        raise StorageError(f"Unsafe storage key segment: {segment!r}")
    return segment


def audio_key(user_id: str, job_id: str, extension: str) -> str:
    """Key for a job's original audio. Uses a generated name, never the user's filename."""
    ext = extension.lower().lstrip(".")
    return f"audio/{_safe(user_id)}/{_safe(job_id)}/original.{_safe(ext)}"


def transcript_prefix(user_id: str, job_id: str) -> str:
    """Key prefix under which a job's exported transcripts are written."""
    return f"transcripts/{_safe(user_id)}/{_safe(job_id)}"


@runtime_checkable
class StorageBackend(Protocol):
    """Minimal storage contract used by the API and workers."""

    name: str

    def put_file(self, src: Path, key: str, move: bool = False) -> None:
        """Store a local file under ``key`` (optionally moving instead of copying)."""
        ...

    def put_bytes(self, data: bytes, key: str) -> None:
        """Store raw bytes under ``key``."""
        ...

    def get_bytes(self, key: str) -> bytes:
        """Read an object fully into memory (use only for small objects such as transcripts)."""
        ...

    def local_path(self, key: str) -> AbstractContextManager[Path]:
        """Context manager yielding a local filesystem path for ``key``.

        Local storage yields the file in place; remote backends would download to a
        temp file and delete it on exit.
        """
        ...

    def exists(self, key: str) -> bool:
        ...

    def delete(self, key: str) -> None:
        """Delete one object (no error if missing)."""
        ...

    def delete_prefix(self, prefix: str) -> None:
        """Delete every object under ``prefix`` (no error if missing)."""
        ...
