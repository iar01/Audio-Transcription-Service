"""Mock transcription engine for fast local testing and demonstration."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from transcribe_pipeline.models import Segment

logger = logging.getLogger(__name__)

DEFAULT_CANNED_SEGMENTS: List[Segment] = [
    Segment(
        start=0.0,
        end=2.45,
        text="Welcome to the audio transcription pipeline demo.",
        confidence=0.98,
    ),
    Segment(
        start=2.45,
        end=5.12,
        text="This sample demonstrates timestamped speech recognition.",
        confidence=0.94,
    ),
    Segment(
        start=5.12,
        end=8.30,
        text="Engineering decisions prioritize clean modular design and streaming.",
        confidence=0.96,
    ),
]

SAMPLE_SENTENCES: List[str] = [
    "Welcome to the audio transcription pipeline demo.",
    "This sample demonstrates timestamped speech recognition.",
    "Engineering decisions prioritize clean modular design and streaming.",
    "Audio preprocessing normalizes any input format into consistent 16kHz mono WAV.",
    "Memory consumption remains strictly flat via lazy segment streaming.",
    "Voice activity detection skips non-speech intervals to optimize throughput.",
    "Sub-second timestamps are calculated and aligned across all segments.",
    "Output exporters format structured JSON, SRT subtitles, and plain text.",
    "FastAPI routes uploaded audio through input validation before transcription.",
    "Long-duration audio is partitioned efficiently across the processing timeline.",
]


def estimate_audio_duration(path: Path) -> float:
    """Estimate audio duration in seconds using WAV headers or file size approximation."""
    suffix = path.suffix.lower()
    if suffix == ".wav":
        try:
            import wave

            with wave.open(str(path), "rb") as w:
                frames = w.getnframes()
                rate = w.getframerate()
                if rate > 0:
                    return round(frames / float(rate), 2)
        except Exception:
            pass

    # Approximation for compressed audio (MP3/M4A/OGG) at standard ~128 kbps (16,000 bytes/sec)
    try:
        size = path.stat().st_size
        if size > 0:
            return round(size / 16000.0, 2)
    except OSError:
        pass

    return 8.30


class MockTranscriberEngine:
    """Mock STT engine returning canned segments without model downloads or network calls."""

    name: str = "mock"
    model_name: str = "mock-base"

    def __init__(
        self,
        canned_segments: Optional[List[Segment]] = None,
        simulate_silence: bool = False,
    ) -> None:
        """Initialize mock engine.

        Args:
            canned_segments: Custom canned segments to return, or default if None.
            simulate_silence: If True, returns empty list simulating a silent clip.
        """
        self._custom_segments = canned_segments
        self.simulate_silence = simulate_silence

    def transcribe(self, audio_path: Path) -> List[Segment]:
        """Return canned segments for the given audio file, scaling to the audio file's duration.

        Args:
            audio_path: Path to the audio file.

        Returns:
            List[Segment]: Transcribed segments.
        """
        path = Path(audio_path)
        logger.info("Mock engine transcribing '%s'...", path.name)

        if self.simulate_silence:
            logger.warning("Mock engine: Silent file or no speech detected in '%s'", path.name)
            return []

        if self._custom_segments is not None:
            return list(self._custom_segments)

        # Estimate file duration to match the input audio duration
        duration = estimate_audio_duration(path)

        # For short clips (under 10s), return the standard 3-segment demo
        if duration <= 10.0:
            for seg in DEFAULT_CANNED_SEGMENTS:
                logger.info("Transcribed segment [%.2fs -> %.2fs]: %s", seg.start, seg.end, seg.text)
            return list(DEFAULT_CANNED_SEGMENTS)

        # For longer audio (e.g. 5 minutes), generate segments spanning the full timeline
        segments: List[Segment] = []
        current_time = 0.0
        sent_idx = 0

        while current_time < duration:
            seg_len = min(4.5 + ((sent_idx * 1.3) % 4.0), duration - current_time)
            seg_end = round(current_time + seg_len, 2)
            text = SAMPLE_SENTENCES[sent_idx % len(SAMPLE_SENTENCES)]
            conf = round(0.92 + ((sent_idx * 0.017) % 0.07), 4)

            seg = Segment(
                start=round(current_time, 2),
                end=seg_end,
                text=text,
                confidence=conf,
            )
            segments.append(seg)
            current_time = seg_end
            sent_idx += 1

        logger.info("Generated %d mock segments spanning %.1fs of audio", len(segments), duration)
        return segments
