"""TranscriberEngine protocol definition."""

from __future__ import annotations

from pathlib import Path
from typing import List, Protocol, runtime_checkable

from transcribe_pipeline.models import Segment


@runtime_checkable
class TranscriberEngine(Protocol):
    """Protocol for speech-to-text engines."""

    name: str
    model_name: str

    def transcribe(self, audio_path: Path) -> List[Segment]:
        """Transcribe audio file at audio_path into timestamped Segment objects.

        Args:
            audio_path: Path to the audio file (typically preprocessed 16kHz mono WAV).

        Returns:
            List[Segment]: Ordered list of transcribed segments.
        """
        ...
