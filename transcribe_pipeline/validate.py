"""Audio input validation module using file inspection and ffprobe."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Set, Union

from transcribe_pipeline.errors import (
    CorruptAudioError,
    EmptyFileError,
    FFmpegNotFoundError,
    FileMissingError,
    FileSizeLimitExceededError,
    UnsupportedFormatError,
)
from transcribe_pipeline.models import AudioMetadata

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS: Set[str] = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
DEFAULT_MAX_FILE_SIZE_BYTES: int = 500 * 1024 * 1024  # 500 MB default limit


def check_ffprobe_available() -> str:
    """Verify ffprobe is installed and return its executable path.

    Raises:
        FFmpegNotFoundError: If ffprobe is not found in system PATH.
    """
    executable = shutil.which("ffprobe")
    if not executable:
        raise FFmpegNotFoundError(
            "ffprobe executable not found in system PATH. Please install FFmpeg."
        )
    return executable


def probe_audio_stream(
    path: Path, ffprobe_binary: str | None = None
) -> AudioMetadata:
    """Inspect audio file with ffprobe to verify valid audio content and extract metadata.

    Args:
        path: Path to the audio file.
        ffprobe_binary: Optional specific path to ffprobe binary.

    Returns:
        AudioMetadata: Extracted technical metadata.

    Raises:
        FFmpegNotFoundError: If ffprobe is not installed.
        CorruptAudioError: If ffprobe cannot parse the audio stream.
    """
    binary = ffprobe_binary or check_ffprobe_available()

    cmd = [
        binary,
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=codec_type,codec_name,sample_rate,channels,duration",
        "-show_entries",
        "format=duration,size",
        "-of",
        "json",
        str(path),
    ]

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except FileNotFoundError as err:
        raise FFmpegNotFoundError(f"ffprobe execution failed: {err}") from err

    if result.returncode != 0:
        err_msg = result.stderr.strip() or "Unknown ffprobe error"
        raise CorruptAudioError(
            f"File '{path.name}' is corrupt or not a valid audio container: {err_msg}"
        )

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as err:
        raise CorruptAudioError(f"Failed to parse ffprobe output for '{path.name}': {err}") from err

    streams = data.get("streams", [])
    if not streams:
        raise CorruptAudioError(
            f"File '{path.name}' contains no valid audio streams."
        )

    audio_stream = streams[0]
    if audio_stream.get("codec_type") != "audio":
        raise CorruptAudioError(
            f"First stream in '{path.name}' is not an audio stream."
        )

    format_data = data.get("format", {})

    # Extract duration (stream duration takes precedence over format duration)
    duration_str = audio_stream.get("duration") or format_data.get("duration")
    try:
        duration = float(duration_str) if duration_str is not None else 0.0
    except (ValueError, TypeError):
        duration = 0.0

    sample_rate_str = audio_stream.get("sample_rate", "0")
    try:
        sample_rate = int(sample_rate_str)
    except (ValueError, TypeError):
        sample_rate = 0

    channels = int(audio_stream.get("channels", 1))
    codec_name = str(audio_stream.get("codec_name", "unknown"))
    size_bytes = int(format_data.get("size", path.stat().st_size))

    return AudioMetadata(
        path=str(path.resolve()),
        duration=duration,
        sample_rate=sample_rate,
        channels=channels,
        codec_name=codec_name,
        size_bytes=size_bytes,
    )


def validate_audio_file(
    path: Union[str, Path],
    max_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES,
    verify_stream: bool = True,
    ffprobe_binary: str | None = None,
) -> AudioMetadata:
    """Validate that the audio file exists, is non-empty, within size limits,
    has a supported extension, and optionally verify audio validity via ffprobe.

    Args:
        path: Path to the audio file.
        max_size_bytes: Configurable maximum allowable file size in bytes.
        verify_stream: Whether to run ffprobe content verification.
        ffprobe_binary: Optional path to ffprobe executable.

    Returns:
        AudioMetadata: Validated audio metadata.

    Raises:
        FileMissingError: If the file does not exist.
        EmptyFileError: If the file is 0 bytes.
        FileSizeLimitExceededError: If file exceeds max_size_bytes.
        UnsupportedFormatError: If file extension is unsupported.
        CorruptAudioError: If stream verification fails or file is corrupt.
        FFmpegNotFoundError: If ffprobe is requested but not found.
    """
    file_path = Path(path)

    # 1. Existence check
    if not file_path.exists() or not file_path.is_file():
        raise FileMissingError(f"Audio file does not exist: '{file_path}'")

    # 2. File size checks
    file_size = file_path.stat().st_size
    if file_size == 0:
        raise EmptyFileError(f"Audio file is empty (0 bytes): '{file_path.name}'")

    if file_size > max_size_bytes:
        raise FileSizeLimitExceededError(
            f"Audio file size ({file_size} bytes) exceeds limit ({max_size_bytes} bytes): '{file_path.name}'"
        )

    # 3. Extension check
    suffix = file_path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise UnsupportedFormatError(
            f"Unsupported audio format '{suffix}' for '{file_path.name}'. Allowed: {allowed}"
        )

    # 4. Stream verification via ffprobe
    if verify_stream:
        return probe_audio_stream(file_path, ffprobe_binary=ffprobe_binary)

    return AudioMetadata(
        path=str(file_path.resolve()),
        duration=0.0,
        sample_rate=0,
        channels=1,
        codec_name=suffix.lstrip("."),
        size_bytes=file_size,
    )
