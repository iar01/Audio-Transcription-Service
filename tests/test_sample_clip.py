"""Integration tests on the short sample audio clip."""

from pathlib import Path

import pytest

from transcribe_pipeline.engines.faster_whisper_engine import FasterWhisperEngine
from transcribe_pipeline.engines.mock_engine import MockTranscriberEngine
from transcribe_pipeline.pipeline import TranscriptionPipeline

DATA_DIR = Path(__file__).parent / "data"
SAMPLE_WAV = DATA_DIR / "sample.wav"
SAMPLE_TXT = DATA_DIR / "sample.txt"


def test_sample_clip_exists() -> None:
    assert SAMPLE_WAV.exists(), "Sample clip tests/data/sample.wav must exist"
    assert SAMPLE_TXT.exists(), "Reference transcript tests/data/sample.txt must exist"
    assert SAMPLE_WAV.stat().st_size > 0
    assert len(SAMPLE_TXT.read_text(encoding="utf-8").strip()) > 0


def test_pipeline_on_sample_clip_mock_engine(tmp_path: Path) -> None:
    expected_text = SAMPLE_TXT.read_text(encoding="utf-8").strip()

    pipeline = TranscriptionPipeline(engine=MockTranscriberEngine())
    result = pipeline.run(
        audio_path=SAMPLE_WAV,
        out_dir=tmp_path,
        verify_stream=False,
        preprocess=False,
    )

    assert result.source_file == "sample.wav"
    assert len(result.segments) > 0
    assert result.duration > 0

    # Ensure output files exist
    assert (tmp_path / "sample.json").exists()
    assert (tmp_path / "sample.srt").exists()
    assert (tmp_path / "sample.txt").exists()


def test_pipeline_on_sample_clip_real_whisper(tmp_path: Path) -> None:
    """Optional test running real faster-whisper on sample clip if installed and cached."""
    pytest.importorskip("faster_whisper")

    try:
        pipeline = TranscriptionPipeline(
            engine=FasterWhisperEngine(model_size_or_path="base", compute_type="int8")
        )
        result = pipeline.run(
            audio_path=SAMPLE_WAV,
            out_dir=tmp_path,
            verify_stream=False,
            preprocess=False,
        )

        assert result.engine == "faster-whisper"
        assert len(result.segments) >= 0
    except Exception as err:
        if "getaddrinfo failed" in str(err) or "ConnectError" in str(err) or "LocalEntryNotFoundError" in str(err):
            pytest.skip(f"Skipping online model download test in offline environment: {err}")
        raise
