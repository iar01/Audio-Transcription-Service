"""Speech-to-text engine using faster-whisper."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, List, Optional

from transcribe_pipeline.errors import EngineError
from transcribe_pipeline.models import Segment

logger = logging.getLogger(__name__)


def _patch_pyav_compatibility() -> None:
    """Ensure compatibility with newer PyAV versions (av >= 14).
    
    faster-whisper passes `metadata_errors='ignore'` to `av.open()`,
    which was deprecated and removed in modern PyAV versions.
    """
    try:
        import av  # type: ignore

        orig_open = av.open
        if not getattr(orig_open, "_metadata_errors_patched", False):
            def safe_open(*args: Any, **kwargs: Any) -> Any:
                kwargs.pop("metadata_errors", None)
                return orig_open(*args, **kwargs)

            safe_open._metadata_errors_patched = True  # type: ignore[attr-defined]
            av.open = safe_open
    except (ImportError, AttributeError):
        pass


# Apply patch upon module import
_patch_pyav_compatibility()


class FasterWhisperEngine:
    """STT engine powered by faster-whisper running quantized on CPU with VAD."""

    name: str = "faster-whisper"

    def __init__(
        self,
        model_size_or_path: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        vad_filter: bool = True,
        whisper_model_instance: Optional[Any] = None,
        cpu_threads: int = 0,
    ) -> None:
        """Initialize faster-whisper engine.

        Args:
            model_size_or_path: Whisper model size (e.g. 'tiny', 'base', 'small', 'medium', 'large-v3').
            device: Compute device ('cpu' or 'cuda').
            compute_type: Quantization type (default: 'int8').
            vad_filter: Enable Silero VAD filter to skip silence.
            whisper_model_instance: Optional pre-initialized WhisperModel (for testing/dependency injection).
            cpu_threads: CPU threads per model (0 = CTranslate2 default). Lower this when
                running several worker processes on one machine to avoid oversubscription.
        """
        self.model_name = model_size_or_path
        self.device = device
        self.compute_type = compute_type
        self.vad_filter = vad_filter
        self.cpu_threads = cpu_threads
        self._model = whisper_model_instance

    def _get_model(self) -> Any:
        """Lazy load WhisperModel instance."""
        if self._model is not None:
            return self._model

        try:
            _patch_pyav_compatibility()
            from faster_whisper import WhisperModel  # type: ignore
        except ImportError as err:
            raise EngineError(
                "The 'faster-whisper' library is not installed. "
                "Install it with: pip install faster-whisper"
            ) from err

        try:
            logger.info(
                "Loading faster-whisper model '%s' (device=%s, compute_type=%s)...",
                self.model_name,
                self.device,
                self.compute_type,
            )
            self._model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
                cpu_threads=self.cpu_threads,
            )
            return self._model
        except Exception as err:
            raise EngineError(
                f"Failed to load faster-whisper model '{self.model_name}': {err}"
            ) from err

    def transcribe(self, audio_path: Path) -> List[Segment]:
        """Transcribe audio file into timestamped Segment objects using lazy generator consumption.

        Args:
            audio_path: Path to preprocessed audio file.

        Returns:
            List[Segment]: Transcribed segments.

        Raises:
            EngineError: If model loading or transcription fails.
        """
        path = Path(audio_path).resolve()
        if not path.exists():
            raise EngineError(f"Audio file not found for transcription: {path}")

        model = self._get_model()

        logger.info(
            "Starting transcription on '%s' with model '%s' (VAD=%s)...",
            path.name,
            self.model_name,
            self.vad_filter,
        )

        try:
            raw_segments, info = model.transcribe(
                str(path),
                vad_filter=self.vad_filter,
            )
        except Exception as err:
            raise EngineError(
                f"faster-whisper failed while transcribing '{path.name}': {err}"
            ) from err

        results: List[Segment] = []

        # Consume the generator lazily so memory stays flat even on long audio files
        for s in raw_segments:
            # Calculate segment confidence: exp(avg_logprob)
            conf: Optional[float] = None
            if getattr(s, "avg_logprob", None) is not None:
                try:
                    raw_conf = math.exp(s.avg_logprob)
                    conf = max(0.0, min(1.0, round(raw_conf, 4)))
                except (OverflowError, ValueError):
                    conf = None

            seg = Segment(
                start=round(float(s.start), 3),
                end=round(float(s.end), 3),
                text=s.text.strip(),
                confidence=conf,
            )
            results.append(seg)
            logger.info("Transcribed segment [%.2fs -> %.2fs]: %s", seg.start, seg.end, seg.text)

        if not results:
            logger.warning("No speech detected or file is silent: '%s'", path.name)

        return results
