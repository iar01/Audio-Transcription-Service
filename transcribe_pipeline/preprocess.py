"""Audio preprocessing module using ffmpeg for audio normalization to 16kHz mono WAV."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Union

from transcribe_pipeline.errors import FFmpegNotFoundError, PreprocessingError

logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE: int = 16000  # 16 kHz standard for Whisper STT
TARGET_CHANNELS: int = 1         # Mono


def check_ffmpeg_available() -> str:
    """Verify ffmpeg is installed and return its executable path.

    Raises:
        FFmpegNotFoundError: If ffmpeg is not found in system PATH.
    """
    executable = shutil.which("ffmpeg")
    if not executable:
        raise FFmpegNotFoundError(
            "ffmpeg executable not found in system PATH. Please install FFmpeg."
        )
    return executable


def convert_audio_to_wav(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    sample_rate: int = TARGET_SAMPLE_RATE,
    channels: int = TARGET_CHANNELS,
    ffmpeg_binary: str | None = None,
) -> Path:
    """Convert an audio file to uncompressed 16-bit PCM WAV at the target sample rate and channels.

    Args:
        input_path: Path to source audio file.
        output_path: Path for output .wav file.
        sample_rate: Target sample rate in Hz (default: 16000).
        channels: Target channel count (1=mono, 2=stereo, default: 1).
        ffmpeg_binary: Optional path to ffmpeg binary.

    Returns:
        Path: Path to the generated WAV file.

    Raises:
        FFmpegNotFoundError: If ffmpeg is not installed.
        PreprocessingError: If ffmpeg conversion fails.
    """
    src = Path(input_path).resolve()
    dst = Path(output_path).resolve()
    binary = ffmpeg_binary or check_ffmpeg_available()

    dst.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        binary,
        "-y",               # Overwrite output without prompting
        "-v",
        "error",            # Quiet non-error output
        "-i",
        str(src),
        "-ar",
        str(sample_rate),   # Resample to target rate (e.g. 16000)
        "-ac",
        str(channels),      # Set channel count (1 for mono)
        "-c:a",
        "pcm_s16le",        # 16-bit PCM standard WAV
        str(dst),
    ]

    logger.debug("Running ffmpeg command: %s", " ".join(cmd))

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except FileNotFoundError as err:
        raise FFmpegNotFoundError(f"ffmpeg execution failed: {err}") from err

    if proc.returncode != 0:
        err_msg = proc.stderr.strip() or "Unknown ffmpeg conversion failure"
        raise PreprocessingError(
            f"Failed to convert '{src.name}' to WAV: {err_msg}"
        )

    if not dst.exists() or dst.stat().st_size == 0:
        raise PreprocessingError(
            f"ffmpeg conversion succeeded but output file is missing or empty: {dst}"
        )

    logger.info("Successfully converted '%s' -> '%s' (16kHz mono WAV)", src.name, dst.name)
    return dst


@contextmanager
def preprocessed_audio(
    input_path: Union[str, Path],
    sample_rate: int = TARGET_SAMPLE_RATE,
    channels: int = TARGET_CHANNELS,
    ffmpeg_binary: str | None = None,
) -> Generator[Path, None, None]:
    """Context manager that normalizes input audio to 16kHz mono WAV in a temporary directory
    and guarantees immediate cleanup of all temporary artifacts upon exit or failure.

    Yields:
        Path: Path to the temporary 16kHz mono WAV file.

    Raises:
        FFmpegNotFoundError: If ffmpeg is not found.
        PreprocessingError: If audio conversion fails.
    """
    src = Path(input_path).resolve()
    temp_dir = tempfile.mkdtemp(prefix="transcribe_pre_")

    try:
        wav_path = Path(temp_dir) / f"{src.stem}_16k_mono.wav"
        convert_audio_to_wav(
            input_path=src,
            output_path=wav_path,
            sample_rate=sample_rate,
            channels=channels,
            ffmpeg_binary=ffmpeg_binary,
        )
        yield wav_path
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        logger.debug("Cleaned up temporary preprocessing directory: %s", temp_dir)
