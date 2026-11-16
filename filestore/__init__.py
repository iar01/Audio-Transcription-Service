"""Persistent file storage for original audio and exported transcripts.

Named ``filestore`` (not ``storage``) so the package doesn't collide with the
default on-disk data directory ``storage/``.
"""

from filestore.base import ObjectNotFoundError, StorageBackend, StorageError, audio_key, transcript_prefix
from filestore.factory import get_storage, reset_storage
from filestore.local import LocalStorage

__all__ = [
    "StorageBackend",
    "StorageError",
    "ObjectNotFoundError",
    "LocalStorage",
    "audio_key",
    "transcript_prefix",
    "get_storage",
    "reset_storage",
]
