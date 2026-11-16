"""SQLAlchemy ORM models for asynchronous transcription jobs.

The database is the source of truth for job/chunk state; Celery task state is
never relied on. Audio and exported transcripts live in file storage and are
referenced by storage *keys* (never absolute paths).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    """Timezone-aware UTC now."""
    return datetime.now(timezone.utc)


def new_job_id() -> str:
    return uuid.uuid4().hex


class JobStatus:
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    TERMINAL = frozenset({COMPLETED, FAILED, CANCELLED})
    ALL = frozenset({QUEUED, PROCESSING, COMPLETED, FAILED, CANCELLED})


class ChunkStatus:
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Base(DeclarativeBase):
    pass


class TranscriptionJob(Base):
    __tablename__ = "transcription_jobs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_job_id)
    user_id: Mapped[str] = mapped_column(String(64), nullable=False, default="anonymous")
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    audio_storage_key: Mapped[str] = mapped_column(String(512), nullable=False)
    transcript_storage_prefix: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False)
    duration: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    status: Mapped[str] = mapped_column(String(16), nullable=False, default=JobStatus.QUEUED)
    progress: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    chunked: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    total_chunks: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    engine: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    model: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    language: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    celery_task_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow
    )

    chunks: Mapped[List["TranscriptionChunk"]] = relationship(
        back_populates="job", cascade="all, delete-orphan", passive_deletes=True,
        order_by="TranscriptionChunk.chunk_index",
    )

    __table_args__ = (
        Index("ix_jobs_status_updated", "status", "updated_at"),
        Index("ix_jobs_created_at", "created_at"),
        Index("ix_jobs_user_id", "user_id"),
    )


class TranscriptionChunk(Base):
    """One application-level slice of a job's audio.

    ``start_time``/``end_time`` are the *nominal* (owned) boundaries used to
    de-duplicate segments; ``audio_start``/``audio_end`` are the actual window
    sent to the engine (nominal range widened by the configured overlap).
    """

    __tablename__ = "transcription_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("transcription_jobs.id", ondelete="CASCADE"), nullable=False
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    start_time: Mapped[float] = mapped_column(Float, nullable=False)
    end_time: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    audio_start: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    audio_end: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    is_full_file: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    status: Mapped[str] = mapped_column(String(16), nullable=False, default=ChunkStatus.PENDING)
    segment_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    claim_token: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow
    )

    job: Mapped[TranscriptionJob] = relationship(back_populates="chunks")

    __table_args__ = (
        UniqueConstraint("job_id", "chunk_index", name="uq_chunk_job_index"),
        Index("ix_chunks_job_status", "job_id", "status"),
        Index("ix_chunks_status_started", "status", "started_at"),
    )


class TranscriptSegment(Base):
    """A timestamped segment with absolute (whole-file) timestamps."""

    __tablename__ = "transcript_segments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("transcription_jobs.id", ondelete="CASCADE"), nullable=False
    )
    chunk_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("transcription_chunks.id", ondelete="CASCADE"), nullable=True
    )
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    start: Mapped[float] = mapped_column(Float, nullable=False)
    end: Mapped[float] = mapped_column(Float, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    # The existing pipeline has no diarization; kept nullable for future use.
    speaker: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    __table_args__ = (
        UniqueConstraint("chunk_id", "seq", name="uq_segment_chunk_seq"),
        Index("ix_segments_job_start", "job_id", "start"),
    )
