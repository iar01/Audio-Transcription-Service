"""Error classification and user-safe error messages.

Internal exception messages can contain filesystem paths or FFmpeg stderr, so
only these generic messages are persisted to the database / returned by the API.
Full details go to the logs.
"""

from __future__ import annotations

from filestore.base import ObjectNotFoundError, StorageError
from transcribe_pipeline.errors import (
    CorruptAudioError,
    EmptyFileError,
    EngineError,
    FFmpegNotFoundError,
    FileMissingError,
    FileSizeLimitExceededError,
    PreprocessingError,
    UnsupportedFormatError,
    ValidationError,
)


class PermanentTaskError(Exception):
    """A failure that retrying will not fix."""


def is_permanent(exc: BaseException) -> bool:
    """Return True for failures that must not be retried."""
    return isinstance(exc, (ValidationError, FFmpegNotFoundError, ObjectNotFoundError, PermanentTaskError))


def public_message(exc: BaseException) -> str:
    """Map an exception to a message that is safe to show users."""
    if isinstance(exc, FileMissingError):
        return "The stored audio file could not be found."
    if isinstance(exc, EmptyFileError):
        return "The audio file is empty."
    if isinstance(exc, FileSizeLimitExceededError):
        return "The audio file exceeds the maximum allowed size."
    if isinstance(exc, UnsupportedFormatError):
        return "Unsupported audio format."
    if isinstance(exc, CorruptAudioError):
        return "The audio file is corrupt or contains no valid audio stream."
    if isinstance(exc, FFmpegNotFoundError):
        return "FFmpeg is not installed on the worker machine."
    if isinstance(exc, PreprocessingError):
        return "Audio preprocessing failed."
    if isinstance(exc, EngineError):
        return "The transcription engine failed."
    if isinstance(exc, ObjectNotFoundError):
        return "The stored audio file could not be found."
    if isinstance(exc, StorageError):
        return "A storage error occurred."
    if isinstance(exc, PermanentTaskError):
        return str(exc) or "The job cannot be processed."
    return "An internal processing error occurred."
