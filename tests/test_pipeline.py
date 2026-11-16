"""Unit and integration tests for the full transcription pipeline."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from transcribe_pipeline.engines.mock_engine import MockTranscriberEngine
from transcribe_pipeline.errors import FileMissingError
from transcribe_pipeline.models import AudioMetadata, Segment, TranscriptionResult
from transcribe_pipeline.pipeline import TranscriptionPipeline, run_pipeline


def test_pipeline_e2e_with_mock_engine(tmp_path: Path) -> None:
    audio_file = tmp_path / "sample.wav"
    audio_file.write_bytes(b"RIFF dummy audio wav bytes")
    out_dir = tmp_path / "out"

    # Mock validate and preprocess to avoid requiring ffmpeg on test runner
    mock_meta = AudioMetadata(
        path=str(audio_file),
        duration=8.30,
        sample_rate=16000,
        channels=1,
        codec_name="pcm_s16le",
        size_bytes=audio_file.stat().st_size,
    )

    with patch("transcribe_pipeline.pipeline.validate_audio_file", return_value=mock_meta), \
         patch("transcribe_pipeline.pipeline.preprocessed_audio") as mock_prep:
        mock_prep.return_value.__enter__.return_value = audio_file

        engine = MockTranscriberEngine()
        pipeline = TranscriptionPipeline(engine=engine)
        result = pipeline.run(audio_path=audio_file, out_dir=out_dir)

        assert isinstance(result, TranscriptionResult)
        assert result.source_file == "sample.wav"
        assert result.engine == "mock"
        assert result.duration == 8.30
        assert len(result.segments) == 3
        assert "Welcome to the audio transcription pipeline demo." in result.full_text

        # Verify export files generated
        assert (out_dir / "sample.json").exists()
        assert (out_dir / "sample.srt").exists()
        assert (out_dir / "sample.txt").exists()


def test_pipeline_missing_file_raises(tmp_path: Path) -> None:
    missing = tmp_path / "not_there.wav"
    pipeline = TranscriptionPipeline(engine=MockTranscriberEngine())
    with pytest.raises(FileMissingError, match="does not exist"):
        pipeline.run(audio_path=missing, verify_stream=False)


def test_run_pipeline_helper(tmp_path: Path) -> None:
    audio = tmp_path / "clip.mp3"
    audio.write_bytes(b"fake audio")

    mock_meta = AudioMetadata(
        path=str(audio),
        duration=5.0,
        sample_rate=16000,
        channels=1,
        codec_name="mp3",
        size_bytes=audio.stat().st_size,
    )

    with patch("transcribe_pipeline.pipeline.validate_audio_file", return_value=mock_meta), \
         patch("transcribe_pipeline.pipeline.preprocessed_audio") as mock_prep:
        mock_prep.return_value.__enter__.return_value = audio

        res = run_pipeline(audio_path=audio, engine_name="mock", verify_stream=False)
        assert res.engine == "mock"
        assert len(res.segments) > 0
