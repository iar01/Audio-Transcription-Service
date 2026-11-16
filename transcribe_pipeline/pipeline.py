"""Core transcription pipeline orchestrator."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, Optional, Union

from transcribe_pipeline.engines.base import TranscriberEngine
from transcribe_pipeline.engines.faster_whisper_engine import FasterWhisperEngine
from transcribe_pipeline.engines.mock_engine import MockTranscriberEngine
from transcribe_pipeline.errors import EngineError, PipelineError
from transcribe_pipeline.exporters import export_all
from transcribe_pipeline.models import AudioMetadata, TranscriptionResult
from transcribe_pipeline.preprocess import preprocessed_audio
from transcribe_pipeline.validate import validate_audio_file

logger = logging.getLogger("transcribe_pipeline.pipeline")


class TranscriptionPipeline:
    """Orchestrates audio validation, ffmpeg preprocessing, transcription, and exporting."""

    def __init__(
        self,
        engine: Optional[TranscriberEngine] = None,
        engine_name: str = "mock",
        model: str = "base",
    ) -> None:
        """Initialize pipeline with specified engine.

        Args:
            engine: Pre-constructed TranscriberEngine instance, or None to construct from name.
            engine_name: 'faster-whisper' or 'mock' (used if engine is None).
            model: Model name/variant to use for faster-whisper (default: 'base').
        """
        if engine is not None:
            self.engine = engine
        elif engine_name == "faster-whisper":
            self.engine = FasterWhisperEngine(model_size_or_path=model)
        elif engine_name == "mock":
            self.engine = MockTranscriberEngine()
        else:
            raise EngineError(f"Unsupported engine type: '{engine_name}'. Choose 'faster-whisper' or 'mock'.")

    def run(
        self,
        audio_path: Union[str, Path],
        out_dir: Optional[Union[str, Path]] = None,
        verify_stream: bool = True,
        max_size_bytes: Optional[int] = None,
        preprocess: bool = True,
    ) -> TranscriptionResult:
        """Execute the end-to-end transcription pipeline with per-stage timing.

        Args:
            audio_path: Path to the input audio file.
            out_dir: Optional directory to export JSON, SRT, and TXT files.
            verify_stream: Whether to run ffprobe validation.
            max_size_bytes: Optional custom file size limit in bytes.

        Returns:
            TranscriptionResult: Structured transcription result with metadata and segments.

        Raises:
            ValidationError: If input validation fails.
            PreprocessingError: If audio conversion fails.
            EngineError: If transcription fails.
            ExportError: If export writing fails.
        """
        start_total = time.perf_counter()
        src_path = Path(audio_path).resolve()

        logger.info("=== Starting Transcription Pipeline for '%s' ===", src_path.name)

        # -------------------------------------------------------------
        # Stage 1: Validation
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        validation_kwargs = {"verify_stream": verify_stream}
        if max_size_bytes is not None:
            validation_kwargs["max_size_bytes"] = max_size_bytes

        metadata: AudioMetadata = validate_audio_file(src_path, **validation_kwargs)
        stage1_time = time.perf_counter() - t0
        logger.info("[Stage 1/3] Validation passed in %.3fs (duration=%.2fs, codec=%s)",
                    stage1_time, metadata.duration, metadata.codec_name)

        # -------------------------------------------------------------
        # Stage 2: Preprocessing (16kHz mono WAV conversion)
        # -------------------------------------------------------------
        if preprocess:
            t1 = time.perf_counter()
            with preprocessed_audio(src_path) as preprocessed_wav:
                stage2_time = time.perf_counter() - t1
                logger.info("[Stage 2/3] Preprocessing (16kHz mono WAV) completed in %.3fs", stage2_time)

                # ---------------------------------------------------------
                # Stage 3: Transcription
                # ---------------------------------------------------------
                t2 = time.perf_counter()
                logger.info("[Stage 3/3] Running transcription engine '%s' (model='%s')...",
                            self.engine.name, self.engine.model_name)
                segments = self.engine.transcribe(preprocessed_wav)
                stage3_time = time.perf_counter() - t2
                logger.info("[Stage 3/3] Transcription completed in %.3fs (%d segments generated)",
                            stage3_time, len(segments))
        else:
            logger.info("[Stage 2/3] Preprocessing skipped (--no-preprocess)")
            t2 = time.perf_counter()
            logger.info("[Stage 3/3] Running transcription engine '%s' (model='%s')...",
                        self.engine.name, self.engine.model_name)
            segments = self.engine.transcribe(src_path)
            stage3_time = time.perf_counter() - t2
            logger.info("[Stage 3/3] Transcription completed in %.3fs (%d segments generated)",
                        stage3_time, len(segments))

        # Build final aggregated result
        duration = metadata.duration if metadata.duration > 0 else (segments[-1].end if segments else 0.0)
        result = TranscriptionResult(
            source_file=src_path.name,
            engine=self.engine.name,
            model=self.engine.model_name,
            language="en",
            duration=duration,
            segments=segments,
        )

        # -------------------------------------------------------------
        # Optional Stage 4: Exporting
        # -------------------------------------------------------------
        if out_dir is not None:
            t3 = time.perf_counter()
            exported = export_all(result, out_dir)
            stage4_time = time.perf_counter() - t3
            logger.info("[Export] Wrote outputs in %.3fs to '%s': %s",
                        stage4_time, out_dir, ", ".join(p.name for p in exported.values()))

        total_time = time.perf_counter() - start_total
        logger.info("=== Pipeline Completed in %.3fs ===", total_time)
        return result


def run_pipeline(
    audio_path: Union[str, Path],
    out_dir: Optional[Union[str, Path]] = None,
    engine_name: str = "mock",
    model: str = "base",
    verify_stream: bool = True,
    preprocess: bool = True,
) -> TranscriptionResult:
    """Convenience helper to initialize and run the pipeline in one call."""
    pipeline = TranscriptionPipeline(engine_name=engine_name, model=model)
    return pipeline.run(
        audio_path=audio_path,
        out_dir=out_dir,
        verify_stream=verify_stream,
        preprocess=preprocess,
    )
