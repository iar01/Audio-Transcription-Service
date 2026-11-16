"""Unit tests for models and custom error hierarchy."""

import pytest

from transcribe_pipeline.errors import (
    CorruptAudioError,
    EmptyFileError,
    FFmpegNotFoundError,
    FileMissingError,
    FileSizeLimitExceededError,
    PipelineError,
    PreprocessingError,
    UnsupportedFormatError,
    ValidationError,
)
from transcribe_pipeline.models import Segment, TranscriptionResult


def test_segment_creation_valid() -> None:
    seg = Segment(start=0.0, end=2.5, text="Hello world", confidence=0.95)
    assert seg.start == 0.0
    assert seg.end == 2.5
    assert seg.text == "Hello world"
    assert seg.confidence == 0.95
    assert seg.duration == 2.5


def test_segment_without_confidence() -> None:
    seg = Segment(start=1.0, end=3.0, text="No confidence score")
    assert seg.confidence is None
    assert seg.duration == 2.0


def test_segment_negative_start_raises() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        Segment(start=-1.0, end=2.0, text="Invalid start")


def test_segment_end_before_start_raises() -> None:
    with pytest.raises(ValueError, match="cannot be earlier than start"):
        Segment(start=5.0, end=2.0, text="Invalid bounds")


def test_segment_to_dict() -> None:
    seg = Segment(start=0.5, end=1.5, text="Dict test", confidence=0.88)
    data = seg.to_dict()
    assert data == {
        "start": 0.5,
        "end": 1.5,
        "text": "Dict test",
        "confidence": 0.88,
    }


def test_transcription_result_full_text_and_dict() -> None:
    s1 = Segment(start=0.0, end=1.5, text="First segment.", confidence=0.9)
    s2 = Segment(start=1.5, end=3.0, text="Second segment.", confidence=0.85)

    res = TranscriptionResult(
        source_file="test.wav",
        engine="mock",
        model="base",
        language="en",
        duration=3.0,
        segments=[s1, s2],
    )

    assert res.full_text == "First segment. Second segment."
    result_dict = res.to_dict()
    assert result_dict["source_file"] == "test.wav"
    assert result_dict["engine"] == "mock"
    assert len(result_dict["segments"]) == 2
    assert result_dict["text"] == "First segment. Second segment."


def test_error_hierarchy() -> None:
    assert issubclass(ValidationError, PipelineError)
    assert issubclass(FileMissingError, ValidationError)
    assert issubclass(EmptyFileError, ValidationError)
    assert issubclass(FileSizeLimitExceededError, ValidationError)
    assert issubclass(UnsupportedFormatError, ValidationError)
    assert issubclass(CorruptAudioError, ValidationError)
    assert issubclass(PreprocessingError, PipelineError)
    assert issubclass(FFmpegNotFoundError, PreprocessingError)
