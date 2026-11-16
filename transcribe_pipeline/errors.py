"""Custom exceptions for the audio transcription pipeline."""

from __future__ import annotations


class PipelineError(Exception):
    """Base exception for all audio transcription pipeline errors."""


class ValidationError(PipelineError):
    """Base class for audio input validation failures."""


class FileMissingError(ValidationError):
    """Raised when the specified audio file path does not exist."""


class EmptyFileError(ValidationError):
    """Raised when the audio file is empty (0 bytes)."""


class FileSizeLimitExceededError(ValidationError):
    """Raised when the audio file exceeds the maximum allowed size limit."""


class UnsupportedFormatError(ValidationError):
    """Raised when the file format or extension is not supported."""


class CorruptAudioError(ValidationError):
    """Raised when ffprobe fails to parse or detects a corrupt audio stream."""


class PreprocessingError(PipelineError):
    """Base class for audio preprocessing failures."""


class FFmpegNotFoundError(PreprocessingError):
    """Raised when ffmpeg or ffprobe executable is not installed or not in PATH."""


class EngineError(PipelineError):
    """Raised when an STT engine fails during model loading or transcription."""


class ExportError(PipelineError):
    """Raised when transcript exporter encounters an error writing outputs."""
