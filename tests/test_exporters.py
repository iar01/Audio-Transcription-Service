"""Unit tests for transcript exporters (SRT, JSON, TXT)."""

import json
from pathlib import Path

import pytest

from transcribe_pipeline.errors import ExportError
from transcribe_pipeline.exporters import (
    export_all,
    export_json,
    export_srt,
    export_txt,
    format_srt_timestamp,
)
from transcribe_pipeline.models import Segment, TranscriptionResult


def test_srt_timestamp_formatting_edge_cases() -> None:
    # Required edge cases in instructions.md
    assert format_srt_timestamp(0.0) == "00:00:00,000"
    assert format_srt_timestamp(59.999) == "00:00:59,999"
    assert format_srt_timestamp(3661.5) == "01:01:01,500"

    # Additional boundary tests
    assert format_srt_timestamp(3599.999) == "00:59:59,999"
    assert format_srt_timestamp(3600.0) == "01:00:00,000"


def test_export_srt(tmp_path: Path) -> None:
    segments = [
        Segment(start=0.0, end=1.5, text="First subtitle line.", confidence=0.99),
        Segment(start=2.0, end=4.123, text="Second subtitle line.", confidence=0.95),
    ]

    out_file = tmp_path / "subs.srt"
    export_srt(segments, out_file)

    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")

    expected = (
        "1\n"
        "00:00:00,000 --> 00:00:01,500\n"
        "First subtitle line.\n\n"
        "2\n"
        "00:00:02,000 --> 00:00:04,123\n"
        "Second subtitle line.\n"
    )
    assert content.strip() == expected.strip()


def test_export_json_structure(tmp_path: Path) -> None:
    segments = [
        Segment(start=0.0, end=2.0, text="Sample text.", confidence=0.9),
    ]
    result = TranscriptionResult(
        source_file="sample.wav",
        engine="mock",
        model="base",
        language="en",
        duration=2.0,
        segments=segments,
    )

    out_file = tmp_path / "transcript.json"
    export_json(result, out_file)

    assert out_file.exists()
    data = json.loads(out_file.read_text(encoding="utf-8"))

    assert data["source_file"] == "sample.wav"
    assert data["engine"] == "mock"
    assert data["model"] == "base"
    assert data["language"] == "en"
    assert data["duration"] == 2.0
    assert data["text"] == "Sample text."
    assert len(data["segments"]) == 1
    assert data["segments"][0]["start"] == 0.0
    assert data["segments"][0]["end"] == 2.0
    assert data["segments"][0]["text"] == "Sample text."
    assert data["segments"][0]["confidence"] == 0.9


def test_export_txt(tmp_path: Path) -> None:
    result = TranscriptionResult(
        source_file="audio.mp3",
        engine="mock",
        model="base",
        language="en",
        duration=5.0,
        segments=[
            Segment(start=0.0, end=2.0, text="Line one."),
            Segment(start=2.0, end=4.5, text="Line two."),
        ],
    )

    out_file = tmp_path / "audio.txt"
    export_txt(result, out_file)

    assert out_file.exists()
    assert out_file.read_text(encoding="utf-8").strip() == "Line one. Line two."


def test_export_all(tmp_path: Path) -> None:
    result = TranscriptionResult(
        source_file="clip.wav",
        engine="mock",
        model="base",
        language="en",
        duration=3.0,
        segments=[Segment(start=0.0, end=3.0, text="Full clip text.", confidence=0.97)],
    )

    out_dir = tmp_path / "outputs"
    written = export_all(result, out_dir, base_name="custom_name")

    assert written["json"].exists()
    assert written["srt"].exists()
    assert written["txt"].exists()

    assert written["json"].name == "custom_name.json"
    assert written["srt"].name == "custom_name.srt"
    assert written["txt"].name == "custom_name.txt"
