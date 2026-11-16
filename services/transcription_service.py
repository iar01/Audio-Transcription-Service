"""Bridge between workers and the existing ``transcribe_pipeline`` package.

* One engine (and therefore one loaded Whisper model) per worker process,
  created lazily and reused for every task that process runs.
* Unchunked jobs go through the existing ``TranscriptionPipeline.run``.
* Chunks are cut + normalized with FFmpeg, transcribed by the same engine and
  shifted to absolute timestamps.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Callable, List, Optional

from chunking.audio_splitter import extract_chunk_wav
from chunking.merger import offset_segments
from config import get_settings
from transcribe_pipeline.engines.base import TranscriberEngine
from transcribe_pipeline.engines.mock_engine import MockTranscriberEngine
from transcribe_pipeline.models import Segment
from transcribe_pipeline.pipeline import TranscriptionPipeline

logger = logging.getLogger(__name__)

_engine_lock = threading.Lock()
_engine: Optional[TranscriberEngine] = None

# Tests may replace this to inject a custom engine into a worker process.
engine_factory: Optional[Callable[[], TranscriberEngine]] = None


def ffmpeg_available() -> bool:
    return bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


def _build_engine() -> TranscriberEngine:
    if engine_factory is not None:
        return engine_factory()

    settings = get_settings()
    choice = settings.transcription_engine
    if choice == "mock":
        logger.info("Using MockTranscriberEngine (TRANSCRIPTION_ENGINE=mock).")
        return MockTranscriberEngine()

    try:
        import faster_whisper  # type: ignore # noqa: F401

        from transcribe_pipeline.engines.faster_whisper_engine import FasterWhisperEngine
    except ImportError:
        if choice == "faster-whisper":
            raise
        logger.warning("faster-whisper not installed; falling back to MockTranscriberEngine.")
        return MockTranscriberEngine()

    logger.info(
        "Using FasterWhisperEngine (model=%s, device=%s, compute_type=%s, cpu_threads=%s, vad=%s)",
        settings.whisper_model, settings.whisper_device, settings.whisper_compute_type,
        settings.whisper_cpu_threads or "default", settings.whisper_vad_filter,
    )
    return FasterWhisperEngine(
        model_size_or_path=settings.whisper_model,
        device=settings.whisper_device,
        compute_type=settings.whisper_compute_type,
        vad_filter=settings.whisper_vad_filter,
        cpu_threads=settings.whisper_cpu_threads,
    )


def get_engine() -> TranscriberEngine:
    """Return this process's shared engine (the Whisper model itself loads lazily on first use)."""
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = _build_engine()
    return _engine


def reset_engine() -> None:
    global _engine
    with _engine_lock:
        _engine = None


def transcribe_full_file(audio_path: Path) -> List[Segment]:
    """Transcribe a whole (short) file using the existing pipeline unchanged."""
    has_ffmpeg = ffmpeg_available()
    pipeline = TranscriptionPipeline(engine=get_engine())
    result = pipeline.run(
        audio_path,
        verify_stream=has_ffmpeg,
        preprocess=has_ffmpeg,
        max_size_bytes=get_settings().max_upload_size_bytes,
    )
    return list(result.segments)


def transcribe_window(audio_path: Path, audio_start: float, audio_end: float) -> List[Segment]:
    """Transcribe ``[audio_start, audio_end)`` of a file; returns absolute-time segments."""
    temp_dir = tempfile.mkdtemp(prefix="volga_chunk_")
    try:
        wav = Path(temp_dir) / "chunk_16k_mono.wav"
        extract_chunk_wav(audio_path, wav, start=audio_start, duration=audio_end - audio_start)
        segments = get_engine().transcribe(wav)
        return offset_segments(segments, audio_start)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
