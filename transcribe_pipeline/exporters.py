"""Transcripts exporter module for JSON, SRT, and TXT formats."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Union

from transcribe_pipeline.errors import ExportError
from transcribe_pipeline.models import Segment, TranscriptionResult

logger = logging.getLogger(__name__)


def format_srt_timestamp(seconds: float) -> str:
    """Format floating point seconds into standard SRT timestamp format: HH:MM:SS,mmm.

    Args:
        seconds: Time in seconds (non-negative).

    Returns:
        str: Timestamp string formatted as HH:MM:SS,mmm.
    """
    total_ms = max(0, int(round(seconds * 1000)))
    hours = total_ms // 3_600_000
    minutes = (total_ms % 3_600_000) // 60_000
    secs = (total_ms % 60_000) // 1_000
    ms = total_ms % 1_000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def export_srt(segments: List[Segment], output_path: Union[str, Path]) -> Path:
    """Export segments to SubRip Subtitle (SRT) format with sequential indices.

    Args:
        segments: List of timestamped Segments.
        output_path: File path for the .srt output.

    Returns:
        Path: Resolved path to the written file.

    Raises:
        ExportError: If file writing fails.
    """
    dest = Path(output_path).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    for idx, seg in enumerate(segments, start=1):
        start_ts = format_srt_timestamp(seg.start)
        end_ts = format_srt_timestamp(seg.end)
        lines.append(f"{idx}")
        lines.append(f"{start_ts} --> {end_ts}")
        lines.append(seg.text.strip())
        lines.append("")  # Empty separator line per SRT specification

    content = "\n".join(lines)
    try:
        dest.write_text(content, encoding="utf-8")
        logger.info("Exported SRT to '%s'", dest)
        return dest
    except OSError as err:
        raise ExportError(f"Failed to write SRT file to '{dest}': {err}") from err


def export_json(result: TranscriptionResult, output_path: Union[str, Path]) -> Path:
    """Export complete transcription result with metadata and segments to JSON.

    Args:
        result: TranscriptionResult instance.
        output_path: File path for the .json output.

    Returns:
        Path: Resolved path to the written file.

    Raises:
        ExportError: If file writing fails.
    """
    dest = Path(output_path).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        data = result.to_dict()
        with open(dest, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info("Exported JSON to '%s'", dest)
        return dest
    except (OSError, TypeError) as err:
        raise ExportError(f"Failed to write JSON file to '{dest}': {err}") from err


def export_txt(
    content_or_result: Union[TranscriptionResult, List[Segment], str],
    output_path: Union[str, Path],
) -> Path:
    """Export plain text transcript to a TXT file.

    Args:
        content_or_result: Text string, TranscriptionResult, or list of Segments.
        output_path: File path for the .txt output.

    Returns:
        Path: Resolved path to the written file.

    Raises:
        ExportError: If file writing fails.
    """
    dest = Path(output_path).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(content_or_result, TranscriptionResult):
        text = content_or_result.full_text
    elif isinstance(content_or_result, list):
        text = " ".join(seg.text.strip() for seg in content_or_result if seg.text.strip())
    else:
        text = str(content_or_result).strip()

    try:
        dest.write_text(text + "\n", encoding="utf-8")
        logger.info("Exported TXT to '%s'", dest)
        return dest
    except OSError as err:
        raise ExportError(f"Failed to write TXT file to '{dest}': {err}") from err


def export_all(
    result: TranscriptionResult,
    output_dir: Union[str, Path],
    base_name: str | None = None,
) -> Dict[str, Path]:
    """Export result into JSON, SRT, and TXT files in the specified output directory.

    Args:
        result: TranscriptionResult instance.
        output_dir: Target output directory.
        base_name: Optional custom base filename (without extension).

    Returns:
        Dict[str, Path]: Mapping of format name to generated file path ('json', 'srt', 'txt').
    """
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = base_name or Path(result.source_file).stem
    json_path = out_dir / f"{stem}.json"
    srt_path = out_dir / f"{stem}.srt"
    txt_path = out_dir / f"{stem}.txt"

    written = {
        "json": export_json(result, json_path),
        "srt": export_srt(result.segments, srt_path),
        "txt": export_txt(result, txt_path),
    }

    return written
