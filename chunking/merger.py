"""Combine per-chunk segments into one transcript with correct timestamps.

Two mechanisms prevent duplicated words at chunk boundaries:

1. **Ownership (at chunk save time)** – every chunk is transcribed with overlap,
   then segments are shifted to absolute time and a segment is kept only by the
   chunk whose nominal range contains the segment's midpoint. Each overlap
   region is therefore owned by exactly one chunk.
2. **Boundary guard (at assembly time)** – where adjacent segments come from
   different chunks, exact duplicate segments are dropped and repeated leading
   words that echo the previous segment's tail are trimmed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

from transcribe_pipeline.models import Segment

_WORD_RE = re.compile(r"[^\w']+", re.UNICODE)
MAX_ECHO_WORDS = 8


def offset_segments(segments: Iterable[Segment], offset: float) -> List[Segment]:
    """Shift chunk-relative segments to absolute timestamps."""
    out: List[Segment] = []
    for seg in segments:
        start = round(max(0.0, seg.start + offset), 3)
        end = round(max(start, seg.end + offset), 3)
        out.append(Segment(start=start, end=end, text=seg.text, confidence=seg.confidence))
    return out


def select_owned_segments(
    segments: Iterable[Segment],
    owned_start: float,
    owned_end: float,
    is_first: bool,
    is_last: bool,
) -> List[Segment]:
    """Keep absolute-time segments whose midpoint falls in the chunk's owned range."""
    lower = float("-inf") if is_first else owned_start
    upper = float("inf") if is_last else owned_end
    kept: List[Segment] = []
    for seg in segments:
        if not seg.text.strip():
            continue
        mid = (seg.start + seg.end) / 2.0
        if lower <= mid < upper:
            kept.append(seg)
    return kept


def _norm_words(text: str) -> List[str]:
    return [w for w in _WORD_RE.split(text.lower()) if w]


@dataclass(frozen=True)
class SourcedSegment:
    """A segment tagged with the chunk it came from (``None`` for unchunked jobs)."""

    segment: Segment
    chunk_index: Optional[int]


def _trim_echo(prev_text: str, text: str) -> str:
    """Remove leading words of ``text`` that repeat the trailing words of ``prev_text``."""
    prev_words = _norm_words(prev_text)
    raw_tokens = text.split()
    cur_words = [_norm_words(t) for t in raw_tokens]
    flat = [w[0] if w else "" for w in cur_words]
    best = 0
    for n in range(min(MAX_ECHO_WORDS, len(prev_words), len(flat)), 1, -1):  # require >= 2 words
        if prev_words[-n:] == flat[:n]:
            best = n
            break
    if best == 0:
        return text
    return " ".join(raw_tokens[best:])


def merge_segments(items: Sequence[SourcedSegment]) -> List[Segment]:
    """Order segments by time and apply the boundary guard between different chunks."""
    ordered = sorted(items, key=lambda s: (s.segment.start, s.segment.end))
    merged: List[Segment] = []
    prev_chunk: Optional[int] = None
    for item in ordered:
        seg = item.segment
        if merged and item.chunk_index is not None and item.chunk_index != prev_chunk:
            prev = merged[-1]
            overlaps = seg.start < prev.end
            if overlaps and _norm_words(seg.text) == _norm_words(prev.text):
                continue  # exact duplicate across the boundary
            trimmed = _trim_echo(prev.text, seg.text)
            if not trimmed.strip():
                continue
            start = seg.start
            if overlaps and prev.end < seg.end:
                start = prev.end  # keep timeline monotonic
            seg = Segment(start=start, end=seg.end, text=trimmed, confidence=seg.confidence)
        merged.append(seg)
        prev_chunk = item.chunk_index
    return merged


def clamp_to_duration(segments: Iterable[Segment], duration: float) -> List[Segment]:
    """Clamp timestamps to the known file duration (engines can overshoot the final frame)."""
    out: List[Segment] = []
    for seg in segments:
        end = min(seg.end, duration)
        start = min(seg.start, end)
        out.append(seg if (start, end) == (seg.start, seg.end)
                   else Segment(start=start, end=end, text=seg.text, confidence=seg.confidence))
    return out
