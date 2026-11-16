"""Unit tests for the CLI interface."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from transcribe_pipeline.cli import main
from transcribe_pipeline.models import AudioMetadata, Segment, TranscriptionResult


def test_cli_help(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "transcribe" in captured.out


def test_cli_transcribe_mock_engine(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    audio = tmp_path / "test.wav"
    audio.write_bytes(b"dummy")
    out_dir = tmp_path / "cli_out"

    mock_result = TranscriptionResult(
        source_file="test.wav",
        engine="mock",
        model="base",
        language="en",
        duration=5.0,
        segments=[Segment(start=0.0, end=5.0, text="CLI test segment.", confidence=0.99)],
    )

    with patch("transcribe_pipeline.cli.TranscriptionPipeline.run", return_value=mock_result):
        exit_code = main([
            "transcribe",
            str(audio),
            "--out-dir",
            str(out_dir),
            "--engine",
            "mock",
        ])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "SUCCESS" in captured.out
        assert "CLI test segment." not in captured.err


def test_cli_transcribe_error_exit_code(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    missing = tmp_path / "does_not_exist.wav"
    exit_code = main([
        "transcribe",
        str(missing),
        "--no-verify-stream",
    ])
    assert exit_code != 0
    captured = capsys.readouterr()
    # Check one-line readable error without stacktrace
    assert "Error:" in captured.err
    assert "Traceback" not in captured.err


def test_cli_transcribe_error_verbose_exit_code(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    missing = tmp_path / "does_not_exist.wav"
    exit_code = main([
        "transcribe",
        str(missing),
        "--no-verify-stream",
        "--verbose",
    ])
    assert exit_code != 0
