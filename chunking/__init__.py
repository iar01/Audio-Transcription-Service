"""Application-level audio chunking and transcript merging."""

from chunking.audio_splitter import ChunkPlan, extract_chunk_wav, plan_chunks, should_chunk
from chunking.merger import (
    SourcedSegment,
    clamp_to_duration,
    merge_segments,
    offset_segments,
    select_owned_segments,
)

__all__ = [
    "ChunkPlan",
    "plan_chunks",
    "should_chunk",
    "extract_chunk_wav",
    "SourcedSegment",
    "merge_segments",
    "clamp_to_duration",
    "offset_segments",
    "select_owned_segments",
]
