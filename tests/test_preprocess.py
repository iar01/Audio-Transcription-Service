"""Unit tests for audio preprocessing and ffmpeg conversion."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from transcribe_pipeline.errors import FFmpegNotFoundError, PreprocessingError
from transcribe_pipeline.preprocess import convert_audio_to_wav, preprocessed_audio


def test_missing_ffmpeg(tmp_path: Path) -> None:
    source = tmp_path / "input.mp3"
    source.write_bytes(b"mp3 data")
    target = tmp_path / "output.wav"

    with patch("shutil.which", return_value=None):
        with pytest.raises(FFmpegNotFoundError, match="ffmpeg executable not found"):
            convert_audio_to_wav(source, target)


def test_ffmpeg_conversion_failure(tmp_path: Path) -> None:
    source = tmp_path / "input.mp3"
    source.write_bytes(b"mp3 data")
    target = tmp_path / "output.wav"

    mock_proc = MagicMock(returncode=1, stderr="Failed to decode stream")
    with patch("shutil.which", return_value="ffmpeg"), patch("subprocess.run", return_value=mock_proc):
        with pytest.raises(PreprocessingError, match="Failed to convert"):
            convert_audio_to_wav(source, target)


def test_ffmpeg_successful_conversion(tmp_path: Path) -> None:
    source = tmp_path / "input.mp3"
    source.write_bytes(b"dummy mp3 data")
    target = tmp_path / "output.wav"

    def side_effect(*args, **kwargs):
        # Simulate ffmpeg generating the output file
        target.write_bytes(b"RIFF dummy 16k mono wav data")
        return MagicMock(returncode=0, stdout="", stderr="")

    with patch("shutil.which", return_value="ffmpeg"), patch("subprocess.run", side_effect=side_effect):
        result = convert_audio_to_wav(source, target, sample_rate=16000, channels=1)
        assert result == target.resolve()
        assert target.exists()
        assert target.stat().st_size > 0


def test_preprocessed_audio_context_manager_cleanup(tmp_path: Path) -> None:
    source = tmp_path / "input.mp3"
    source.write_bytes(b"dummy mp3")

    temp_wav_holder: list[Path] = []

    def mock_convert(input_path, output_path, **kwargs):
        Path(output_path).write_bytes(b"RIFF wav content")
        return Path(output_path)

    with patch("transcribe_pipeline.preprocess.convert_audio_to_wav", side_effect=mock_convert):
        with preprocessed_audio(source) as wav_path:
            temp_wav_holder.append(wav_path)
            assert wav_path.exists()
            assert wav_path.parent.exists()

    # Verify temp file and parent temp directory are cleanly removed after exit
    assert not temp_wav_holder[0].exists()
    assert not temp_wav_holder[0].parent.exists()


def test_preprocessed_audio_cleanup_on_exception(tmp_path: Path) -> None:
    source = tmp_path / "input.mp3"
    source.write_bytes(b"dummy mp3")

    temp_wav_holder: list[Path] = []

    def mock_convert(input_path, output_path, **kwargs):
        Path(output_path).write_bytes(b"RIFF wav content")
        return Path(output_path)

    with patch("transcribe_pipeline.preprocess.convert_audio_to_wav", side_effect=mock_convert):
        with pytest.raises(RuntimeError, match="Simulated crash"):
            with preprocessed_audio(source) as wav_path:
                temp_wav_holder.append(wav_path)
                assert wav_path.exists()
                raise RuntimeError("Simulated crash")

    # Temp artifacts must still be cleaned up on failure
    assert not temp_wav_holder[0].exists()
    assert not temp_wav_holder[0].parent.exists()
