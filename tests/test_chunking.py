"""Chunk planning, FFmpeg chunk extraction and transcript merging."""

from __future__ import annotations

import wave
from pathlib import Path

import pytest

from chunking.audio_splitter import extract_chunk_wav, plan_chunks, should_chunk
from chunking.merger import (
    SourcedSegment,
    clamp_to_duration,
    merge_segments,
    offset_segments,
    select_owned_segments,
)
from tests.conftest import make_wav, requires_ffmpeg
from transcribe_pipeline.models import Segment


def test_should_chunk_threshold() -> None:
    assert should_chunk(601, 600)
    assert not should_chunk(600, 600)
    assert not should_chunk(None, 600)
    assert not should_chunk(5000, 600, enabled=False)


def test_plan_chunks_boundaries_and_overlap() -> None:
    plans = plan_chunks(960, chunk_duration=300, overlap=3)
    assert [(p.start, p.end) for p in plans] == [(0, 300), (300, 600), (600, 960)]  # 60s tail folded
    assert plans[0].audio_start == 0 and plans[0].audio_end == 303
    assert plans[1].audio_start == 297 and plans[1].audio_end == 603
    assert plans[-1].audio_end == 960  # clamped to file end


def test_plan_chunks_keeps_large_tail() -> None:
    plans = plan_chunks(1100, chunk_duration=300, overlap=3)
    assert [(p.start, p.end) for p in plans][-1] == (900, 1100)
    assert len(plans) == 4


def test_plan_chunks_nominal_ranges_cover_file_without_gaps() -> None:
    plans = plan_chunks(3725.4, chunk_duration=300, overlap=5)
    assert plans[0].start == 0 and plans[-1].end == 3725.4
    for a, b in zip(plans, plans[1:]):
        assert a.end == b.start


def test_plan_chunks_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        plan_chunks(0, 300, 3)


def test_offset_and_ownership_preserve_absolute_timestamps() -> None:
    # Chunk 1 nominal [300, 600) transcribed from window [297, 603)
    relative = [Segment(0.5, 2.5, "tail of previous"), Segment(3.2, 8.0, "first owned"),
                Segment(302.0, 305.0, "mostly in next chunk")]
    absolute = offset_segments(relative, 297.0)
    assert absolute[1].start == pytest.approx(300.2) and absolute[1].end == pytest.approx(305.0)
    owned = select_owned_segments(absolute, 300, 600, is_first=False, is_last=False)
    # midpoints 298.5 (previous chunk) and 600.5 (next chunk) are excluded
    assert [s.text for s in owned] == ["first owned"]


def test_overlap_region_owned_by_exactly_one_chunk() -> None:
    # The same utterance seen by both chunks around the 300s boundary
    seen_by_0 = [Segment(298.0, 301.0, "boundary words")]  # midpoint 299.5 -> chunk 0
    seen_by_1 = [Segment(298.1, 301.1, "boundary words")]  # midpoint 299.6 -> chunk 0 as well
    kept0 = select_owned_segments(seen_by_0, 0, 300, is_first=True, is_last=False)
    kept1 = select_owned_segments(seen_by_1, 300, 600, is_first=False, is_last=True)
    assert len(kept0) + len(kept1) == 1


def test_merge_drops_exact_duplicates_and_trims_echoed_words() -> None:
    items = [
        SourcedSegment(Segment(0.0, 4.0, "the quick brown fox jumps"), 0),
        SourcedSegment(Segment(3.5, 4.2, "The quick brown fox jumps"), 1),  # duplicate across boundary
        SourcedSegment(Segment(4.2, 8.0, "fox jumps over the lazy dog"), 1),
    ]
    merged = merge_segments(items)
    assert [s.text for s in merged] == ["the quick brown fox jumps", "over the lazy dog"]
    assert all(b.start >= a.start for a, b in zip(merged, merged[1:]))


def test_merge_keeps_same_chunk_segments_untouched() -> None:
    items = [SourcedSegment(Segment(0, 1, "yes yes"), 0), SourcedSegment(Segment(1, 2, "yes yes"), 0)]
    assert len(merge_segments(items)) == 2


def test_clamp_to_duration() -> None:
    out = clamp_to_duration([Segment(9.0, 12.0, "x"), Segment(13.0, 14.0, "y")], 10.0)
    assert out[0].end == 10.0 and out[1].start == 10.0 and out[1].end == 10.0


@requires_ffmpeg
def test_extract_chunk_wav_duration(tmp_path: Path) -> None:
    src = make_wav(tmp_path / "long.wav", 12.0, rate=22050)
    out = extract_chunk_wav(src, tmp_path / "c.wav", start=4.0, duration=5.0)
    with wave.open(str(out), "rb") as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1
        assert w.getnframes() / w.getframerate() == pytest.approx(5.0, abs=0.05)
