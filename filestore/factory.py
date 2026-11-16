"""Construct the configured storage backend."""

from __future__ import annotations

import threading
from typing import Optional

from config import get_settings
from filestore.base import StorageBackend, StorageError
from filestore.local import LocalStorage

_lock = threading.Lock()
_storage: Optional[StorageBackend] = None


def get_storage() -> StorageBackend:
    """Return the process-wide storage backend selected by ``STORAGE_BACKEND``."""
    global _storage
    if _storage is None:
        with _lock:
            if _storage is None:
                settings = get_settings()
                if settings.storage_backend == "local":
                    _storage = LocalStorage(settings.storage_path / "files")
                elif settings.storage_backend in {"s3", "minio"}:
                    # The interface (keys + local_path context manager) is designed for an
                    # S3/MinIO backend, but it is not implemented yet.
                    raise StorageError(
                        "STORAGE_BACKEND=s3/minio is not implemented yet; use STORAGE_BACKEND=local."
                    )
                else:
                    raise StorageError(f"Unknown STORAGE_BACKEND: {settings.storage_backend!r}")
    return _storage


def reset_storage() -> None:
    global _storage
    with _lock:
        _storage = None
