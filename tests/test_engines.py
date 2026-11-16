"""Unit tests for transcription engine implementations and protocol conformance."""

import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from transcribe_pipeline.engines.base import TranscriberEngine
from transcribe_pipeline.engines.faster_whisper_engine import FasterWhisperEngine
from transcribe_pipeline.engines.mock_engine import MockTranscriberEngine
from transcribe_pipeline.errors import EngineError
from transcribe_pipeline.models import Segment


def test_mock_engine_protocol_conformance() -> None:
    engine = MockTranscriberEngine()
    assert isinstance(engine, TranscriberEngine)
    assert engine.name == "mock"
    assert engine.model_name == "mock-base"


def test_mock_engine_transcribe(tmp_path: Path) -> None:
    audio = tmp_path / "test.wav"
    audio.write_bytes(b"dummy")

    engine = MockTranscriberEngine()
    segments = engine.transcribe(audio)

    assert len(segments) == 3
    assert segments[0].start == 0.0
    assert segments[0].end == 2.45
    assert segments[0].confidence == 0.98


def test_mock_engine_silence(tmp_path: Path) -> None:
    audio = tmp_path / "silent.wav"
    audio.write_bytes(b"silent")

    engine = MockTranscriberEngine(simulate_silence=True)
    segments = engine.transcribe(audio)
    assert segments == []


def test_mock_engine_custom_segments(tmp_path: Path) -> None:
    audio = tmp_path / "custom.wav"
    audio.write_bytes(b"audio")

    custom = [Segment(start=1.0, end=2.0, text="Custom test", confidence=0.8)]
    engine = MockTranscriberEngine(canned_segments=custom)
    segments = engine.transcribe(audio)
    assert segments == custom


def test_faster_whisper_engine_missing_import(tmp_path: Path) -> None:
    audio = tmp_path / "test.wav"
    audio.write_bytes(b"audio")

    engine = FasterWhisperEngine()
    with patch.dict("sys.modules", {"faster_whisper": None}):
        with pytest.raises(EngineError, match="faster-whisper.*not installed"):
            engine.transcribe(audio)


def test_faster_whisper_engine_transcribe_generator_and_confidence(tmp_path: Path) -> None:
    audio = tmp_path / "test.wav"
    audio.write_bytes(b"audio")

    # Mock raw segment objects returned by faster-whisper
    class MockRawSegment:
        def __init__(self, start, end, text, avg_logprob):
            self.start = start
            self.end = end
            self.text = text
            self.avg_logprob = avg_logprob

    raw_items = [
        MockRawSegment(0.0, 2.5, " Hello world! ", -0.05),
        MockRawSegment(2.5, 4.8, " Second chunk. ", -0.2),
    ]

    mock_whisper_model = MagicMock()
    # Emulate lazy generator return
    mock_whisper_model.transcribe.return_value = (iter(raw_items), MagicMock(language="en"))

    engine = FasterWhisperEngine(whisper_model_instance=mock_whisper_model)
    assert isinstance(engine, TranscriberEngine)

    segments = engine.transcribe(audio)

    assert len(segments) == 2
    assert segments[0].start == 0.0
    assert segments[0].end == 2.5
    assert segments[0].text == "Hello world!"
    # confidence = exp(avg_logprob)
    expected_conf_0 = round(math.exp(-0.05), 4)
    assert segments[0].confidence == expected_conf_0

    assert segments[1].start == 2.5
    assert segments[1].end == 4.8
    assert segments[1].text == "Second chunk."
    expected_conf_1 = round(math.exp(-0.2), 4)
    assert segments[1].confidence == expected_conf_1


def test_faster_whisper_engine_silent_file(tmp_path: Path) -> None:
    audio = tmp_path / "silent.wav"
    audio.write_bytes(b"silent")

    mock_whisper_model = MagicMock()
    mock_whisper_model.transcribe.return_value = (iter([]), MagicMock(language="en"))

    engine = FasterWhisperEngine(whisper_model_instance=mock_whisper_model)
    segments = engine.transcribe(audio)
    assert segments == []
