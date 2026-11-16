"""Data models for audio transcription pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class Segment:
    """A timestamped segment of transcribed speech.

    Attributes:
        start: Start time of segment in seconds.
        end: End time of segment in seconds.
        text: Transcribed text content.
        confidence: Average confidence score in range [0.0, 1.0], or None if unavailable.
    """

    start: float
    end: float
    text: str
    confidence: Optional[float] = None

    def __post_init__(self) -> None:
        """Validate segment time bounds."""
        if self.start < 0:
            raise ValueError(f"Segment start time must be non-negative: {self.start}")
        if self.end < self.start:
            raise ValueError(f"Segment end time ({self.end}) cannot be earlier than start ({self.start})")

    @property
    def duration(self) -> float:
        """Return segment duration in seconds."""
        return self.end - self.start

    def to_dict(self) -> Dict[str, Any]:
        """Convert segment to dictionary representation."""
        return asdict(self)


@dataclass(frozen=True)
class AudioMetadata:
    """Technical metadata extracted from an audio file via ffprobe.

    Attributes:
        path: Path to the audio file.
        duration: Duration of the audio in seconds.
        sample_rate: Audio sample rate in Hz.
        channels: Number of audio channels.
        codec_name: Detected audio codec (e.g. 'pcm_s16le', 'mp3', 'aac').
        size_bytes: File size in bytes.
    """

    path: str
    duration: float
    sample_rate: int
    channels: int
    codec_name: str
    size_bytes: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary representation."""
        return asdict(self)


@dataclass
class TranscriptionResult:
    """Aggregated transcription result including metadata and segment list.

    Attributes:
        source_file: Original audio file path or filename.
        engine: Identifier of engine used (e.g. 'faster-whisper', 'mock').
        model: Model name/variant (e.g. 'base', 'int8').
        language: Detected or configured spoken language code.
        duration: Total audio duration in seconds.
        segments: List of timestamped Segments.
    """

    source_file: str
    engine: str
    model: str
    language: str
    duration: float
    segments: List[Segment] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        """Combine all segment text into a single coherent string."""
        return " ".join(seg.text.strip() for seg in self.segments if seg.text.strip())

    def to_dict(self) -> Dict[str, Any]:
        """Convert complete result and nested segments to dictionary."""
        return {
            "source_file": self.source_file,
            "engine": self.engine,
            "model": self.model,
            "language": self.language,
            "duration": self.duration,
            "text": self.full_text,
            "segments": [seg.to_dict() for seg in self.segments],
        }
