"""Unit tests for audio input validation module."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from transcribe_pipeline.errors import (
    CorruptAudioError,
    EmptyFileError,
    FFmpegNotFoundError,
    FileMissingError,
    FileSizeLimitExceededError,
    UnsupportedFormatError,
)
from transcribe_pipeline.validate import (
    probe_audio_stream,
    validate_audio_file,
)


def test_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "non_existent.wav"
    with pytest.raises(FileMissingError, match="does not exist"):
        validate_audio_file(missing, verify_stream=False)


def test_empty_file(tmp_path: Path) -> None:
    empty = tmp_path / "empty.wav"
    empty.write_bytes(b"")
    with pytest.raises(EmptyFileError, match="empty"):
        validate_audio_file(empty, verify_stream=False)


def test_file_size_limit_exceeded(tmp_path: Path) -> None:
    sample = tmp_path / "toolarge.wav"
    sample.write_bytes(b"x" * 1024)
    with pytest.raises(FileSizeLimitExceededError, match="exceeds limit"):
        validate_audio_file(sample, max_size_bytes=512, verify_stream=False)


def test_unsupported_format(tmp_path: Path) -> None:
    text_file = tmp_path / "notes.txt"
    text_file.write_text("not audio", encoding="utf-8")
    with pytest.raises(UnsupportedFormatError, match="Unsupported audio format"):
        validate_audio_file(text_file, verify_stream=False)

    mp4_video = tmp_path / "video.mp4"
    mp4_video.write_bytes(b"dummy")
    with pytest.raises(UnsupportedFormatError, match="Unsupported audio format"):
        validate_audio_file(mp4_video, verify_stream=False)


def test_ffprobe_missing(tmp_path: Path) -> None:
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"fake wav data")

    with patch("shutil.which", return_value=None):
        with pytest.raises(FFmpegNotFoundError, match="ffprobe executable not found"):
            validate_audio_file(audio, verify_stream=True)


def test_corrupt_file_ffprobe_non_zero_exit(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.wav"
    corrupt.write_bytes(b"junk audio data")

    mock_proc = MagicMock(returncode=1, stderr="Invalid data found when processing input")
    with patch("shutil.which", return_value="ffprobe"), patch("subprocess.run", return_value=mock_proc):
        with pytest.raises(CorruptAudioError, match="corrupt or not a valid audio container"):
            validate_audio_file(corrupt, verify_stream=True)


def test_corrupt_file_no_audio_streams(tmp_path: Path) -> None:
    no_audio = tmp_path / "noaudio.wav"
    no_audio.write_bytes(b"data")

    mock_json = json.dumps({"streams": [], "format": {"duration": "10.0"}})
    mock_proc = MagicMock(returncode=0, stdout=mock_json, stderr="")
    with patch("shutil.which", return_value="ffprobe"), patch("subprocess.run", return_value=mock_proc):
        with pytest.raises(CorruptAudioError, match="contains no valid audio streams"):
            validate_audio_file(no_audio, verify_stream=True)


def test_valid_file_with_ffprobe_metadata(tmp_path: Path) -> None:
    valid_wav = tmp_path / "good.wav"
    valid_wav.write_bytes(b"RIFF dummy audio bytes")

    ffprobe_payload = {
        "streams": [
            {
                "codec_type": "audio",
                "codec_name": "pcm_s16le",
                "sample_rate": "44100",
                "channels": 2,
                "duration": "12.345",
            }
        ],
        "format": {
            "duration": "12.345",
            "size": "22",
        },
    }
    mock_proc = MagicMock(returncode=0, stdout=json.dumps(ffprobe_payload), stderr="")
    with patch("shutil.which", return_value="ffprobe"), patch("subprocess.run", return_value=mock_proc):
        meta = validate_audio_file(valid_wav, verify_stream=True)
        assert meta.codec_name == "pcm_s16le"
        assert meta.sample_rate == 44100
        assert meta.channels == 2
        assert meta.duration == 12.345
