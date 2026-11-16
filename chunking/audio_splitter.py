"""Application-level chunk planning and FFmpeg chunk extraction.

faster-whisper already segments audio internally (30 s windows + Silero VAD).
Application-level chunking is an *additional* reliability layer: each chunk is
an independent unit of work that can be retried and checkpointed on its own.
It is only used for recordings longer than ``CHUNKING_THRESHOLD``.
"""

from __future__ import annotations

import logging
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

from transcribe_pipeline.errors import FFmpegNotFoundError, PreprocessingError
from transcribe_pipeline.preprocess import TARGET_CHANNELS, TARGET_SAMPLE_RATE, check_ffmpeg_available

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChunkPlan:
    """Planned chunk.

    ``start``/``end`` are nominal (owned) boundaries; ``audio_start``/``audio_end``
    is the window actually transcribed (nominal range ± overlap, clamped to the file).
    """

    index: int
    start: float
    end: float
    audio_start: float
    audio_end: float


def should_chunk(duration: Optional[float], threshold: int, enabled: bool = True) -> bool:
    """Return True if a recording of ``duration`` seconds should be split."""
    return bool(enabled and duration and duration > threshold)


def plan_chunks(
    duration: float,
    chunk_duration: float,
    overlap: float,
    min_tail_ratio: float = 0.25,
) -> List[ChunkPlan]:
    """Split ``[0, duration]`` into nominal chunks with symmetric overlap windows.

    A final remainder shorter than ``min_tail_ratio * chunk_duration`` is folded into
    the previous chunk to avoid tiny chunks.
    """
    if duration <= 0:
        raise ValueError("duration must be positive")
    if chunk_duration <= 0:
        raise ValueError("chunk_duration must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")

    count = max(1, math.ceil(duration / chunk_duration))
    if count > 1 and (duration - (count - 1) * chunk_duration) < min_tail_ratio * chunk_duration:
        count -= 1

    plans: List[ChunkPlan] = []
    for i in range(count):
        start = i * chunk_duration
        end = duration if i == count - 1 else (i + 1) * chunk_duration
        plans.append(
            ChunkPlan(
                index=i,
                start=round(start, 3),
                end=round(end, 3),
                audio_start=round(max(0.0, start - overlap), 3),
                audio_end=round(min(duration, end + overlap), 3),
            )
        )
    return plans


def extract_chunk_wav(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    start: float,
    duration: float,
    ffmpeg_binary: Optional[str] = None,
) -> Path:
    """Cut ``[start, start+duration)`` from the source and normalize it to 16 kHz mono WAV.

    Uses the same output format as ``transcribe_pipeline.preprocess`` so the engine
    always receives identical input.
    """
    src = Path(input_path).resolve()
    dst = Path(output_path).resolve()
    binary = ffmpeg_binary or check_ffmpeg_available()
    dst.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        binary, "-y", "-v", "error",
        "-ss", f"{max(0.0, start):.3f}",
        "-i", str(src),
        "-t", f"{max(0.001, duration):.3f}",
        "-ar", str(TARGET_SAMPLE_RATE),
        "-ac", str(TARGET_CHANNELS),
        "-c:a", "pcm_s16le",
        str(dst),
    ]
    logger.debug("Extracting chunk: %s", " ".join(cmd))
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    except FileNotFoundError as err:
        raise FFmpegNotFoundError(f"ffmpeg execution failed: {err}") from err

    if proc.returncode != 0:
        raise PreprocessingError(
            f"Failed to extract chunk at {start:.2f}s from '{src.name}': {proc.stderr.strip() or 'unknown error'}"
        )
    if not dst.exists() or dst.stat().st_size == 0:
        raise PreprocessingError(f"Chunk extraction produced no output for '{src.name}' at {start:.2f}s")
    return dst
