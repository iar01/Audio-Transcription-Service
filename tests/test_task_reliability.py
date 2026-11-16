"""Retry, idempotency, duplicate delivery, crash recovery and retention for Celery tasks.

Runs tasks in eager mode against a real SQLite database and the mock engine.
Chunk failures are injected by wrapping ``transcription_service.transcribe_window``.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Callable, Dict, List

import pytest
from sqlalchemy import func, select, update

from database.models import ChunkStatus, JobStatus, TranscriptionChunk, TranscriptionJob, TranscriptSegment, utcnow
from database.session import session_scope
from filestore import audio_key, get_storage
from services import job_service, transcription_service
from tests.conftest import make_wav, requires_ffmpeg
from transcribe_pipeline.errors import CorruptAudioError, EngineError

pytestmark = requires_ffmpeg  # chunk extraction needs ffmpeg


def _create_job(tmp_path: Path, seconds: float, job_id: str = "job") -> str:
    src = make_wav(tmp_path / f"{job_id}.wav", seconds)
    key = audio_key("anonymous", job_id, "wav")
    get_storage().put_file(src, key)
    job_service.create_job(job_id, f"{job_id}.wav", key, src.stat().st_size, seconds)
    return job_id


def _instrument(monkeypatch: pytest.MonkeyPatch, fail: Callable[[float, int], Exception | None]) -> Dict[float, int]:
    """Wrap chunk transcription; ``fail(audio_start, attempt)`` returns an exception to raise or None."""
    calls: Dict[float, int] = {}
    original = transcription_service.transcribe_window

    def wrapped(path, audio_start, audio_end):  # type: ignore[no-untyped-def]
        calls[audio_start] = calls.get(audio_start, 0) + 1
        exc = fail(audio_start, calls[audio_start])
        if exc is not None:
            raise exc
        return original(path, audio_start, audio_end)

    monkeypatch.setattr(transcription_service, "transcribe_window", wrapped)
    return calls


def _segment_count(job_id: str) -> int:
    with session_scope() as s:
        return int(s.scalar(select(func.count()).select_from(TranscriptSegment)
                            .where(TranscriptSegment.job_id == job_id)) or 0)


def _chunks(job_id: str) -> List[TranscriptionChunk]:
    with session_scope() as s:
        return list(s.scalars(select(TranscriptionChunk).where(TranscriptionChunk.job_id == job_id)
                              .order_by(TranscriptionChunk.chunk_index)).all())


def test_transient_chunk_failure_is_retried_without_redoing_other_chunks(volga_env, tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from workers.transcription_tasks import enqueue_job

    job_id = _create_job(tmp_path, 35.0)
    calls = _instrument(monkeypatch, lambda start, n: EngineError("boom") if start == 18.0 and n <= 2 else None)
    enqueue_job(job_id)

    snap = job_service.job_snapshot(job_id)
    assert snap["status"] == "completed"
    assert calls == {0.0: 1, 8.0: 1, 18.0: 3, 28.0: 1}  # only the failing chunk was repeated
    assert [c.retry_count for c in _chunks(job_id)] == [0, 0, 2, 0]


def test_exhausted_retries_fail_job_but_keep_completed_chunks_then_retry_only_failed(volga_env, tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from workers.transcription_tasks import enqueue_job

    job_id = _create_job(tmp_path, 35.0)
    broken = {"on": True}
    calls = _instrument(monkeypatch, lambda start, n: EngineError("down") if start == 8.0 and broken["on"] else None)
    enqueue_job(job_id)

    snap = job_service.job_snapshot(job_id)
    assert snap["status"] == "failed" and snap["can_retry"]
    assert snap["chunks"] == {"total": 4, "completed": 3, "failed": 1, "processing": 0, "pending": 0}
    assert calls[8.0] == 4  # 1 attempt + CELERY_MAX_RETRIES (3)
    assert "down" not in snap["error"]  # internal exception text is not exposed
    segments_before = _segment_count(job_id)
    assert segments_before > 0  # completed work persisted despite the failure

    broken["on"] = False
    assert job_service.prepare_retry(job_id)
    enqueue_job(job_id)
    assert job_service.job_snapshot(job_id)["status"] == "completed"
    assert calls == {0.0: 1, 8.0: 5, 18.0: 1, 28.0: 1}  # completed chunks were not retranscribed


def test_permanent_errors_are_not_retried(volga_env, tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from workers.transcription_tasks import enqueue_job

    job_id = _create_job(tmp_path, 35.0)
    calls = _instrument(monkeypatch, lambda start, n: CorruptAudioError("/secret/path bad") if start == 0.0 else None)
    enqueue_job(job_id)
    snap = job_service.job_snapshot(job_id)
    assert calls[0.0] == 1
    assert snap["status"] == "failed"
    chunk0 = _chunks(job_id)[0]
    assert chunk0.status == ChunkStatus.FAILED and "/secret/path" not in (chunk0.error_message or "")


def test_missing_audio_fails_job_permanently(volga_env, tmp_path) -> None:  # type: ignore[no-untyped-def]
    from workers.transcription_tasks import enqueue_job

    job_service.create_job("ghost", "ghost.wav", audio_key("anonymous", "ghost", "wav"), 10, 3.0)
    enqueue_job("ghost")
    snap = job_service.job_snapshot("ghost")
    assert snap["status"] == "failed" and snap["error"] == "The stored audio file could not be found."


def test_duplicate_delivery_is_idempotent(volga_env, tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from workers.transcription_tasks import enqueue_job, finalize_job, transcribe_chunk, transcribe_job

    job_id = _create_job(tmp_path, 35.0)
    calls = _instrument(monkeypatch, lambda start, n: None)
    enqueue_job(job_id)
    assert job_service.job_snapshot(job_id)["status"] == "completed"
    n_segments = _segment_count(job_id)

    # Celery delivers everything a second time
    transcribe_job.apply(args=[job_id])
    for chunk in _chunks(job_id):
        assert transcribe_chunk.apply(args=[chunk.id]).result["skipped"] is True
    assert finalize_job.apply(args=[job_id]).result["skipped"] is True

    assert _segment_count(job_id) == n_segments
    assert len(_chunks(job_id)) == 4
    assert all(v == 1 for v in calls.values())


def test_concurrent_claims_only_one_wins(volga_env, tmp_path) -> None:  # type: ignore[no-untyped-def]
    job_id = _create_job(tmp_path, 35.0)
    assert job_service.claim_job(job_id) is not None
    chunks = job_service.ensure_chunks(job_id, 35.0)
    assert job_service.ensure_chunks(job_id, 35.0)[0].id == chunks[0].id  # planning is idempotent
    first = job_service.claim_chunk(chunks[0].id)
    second = job_service.claim_chunk(chunks[0].id)
    assert first is not None and second is None


def test_stale_reclaim_prevents_duplicate_results(volga_env, tmp_path) -> None:  # type: ignore[no-untyped-def]
    from transcribe_pipeline.models import Segment

    job_id = _create_job(tmp_path, 35.0)
    job_service.claim_job(job_id)
    chunk_id = job_service.ensure_chunks(job_id, 35.0)[0].id
    slow_worker = job_service.claim_chunk(chunk_id)
    assert slow_worker is not None

    # The slow worker looks dead: its claim becomes stale and another worker takes over.
    with session_scope() as s:
        s.execute(update(TranscriptionChunk).where(TranscriptionChunk.id == chunk_id)
                  .values(started_at=utcnow() - timedelta(hours=2)))
    new_worker = job_service.claim_chunk(chunk_id)
    assert new_worker is not None

    segs = [Segment(0.0, 1.0, "hello")]
    assert job_service.complete_chunk(new_worker, segs) is True
    assert job_service.complete_chunk(slow_worker, segs) is False  # stale result discarded
    assert _segment_count(job_id) == 1


def test_recovery_after_worker_crash(volga_env, tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Simulate a crash: job processing, one chunk done, one stuck 'processing', the rest pending."""
    from workers import transcription_tasks as tasks

    job_id = _create_job(tmp_path, 35.0)
    calls = _instrument(monkeypatch, lambda start, n: None)

    # Run the first chunk normally, then "crash" while the second is in progress.
    job_service.claim_job(job_id)
    chunks = job_service.ensure_chunks(job_id, 35.0)
    tasks.transcribe_chunk.run(chunks[0].id)
    assert job_service.claim_chunk(chunks[1].id) is not None  # worker died holding this claim
    old = utcnow() - timedelta(hours=2)
    with session_scope() as s:
        s.execute(update(TranscriptionChunk).where(TranscriptionChunk.job_id == job_id)
                  .values(updated_at=old))
        s.execute(update(TranscriptionChunk).where(TranscriptionChunk.id == chunks[1].id).values(started_at=old))

    summary = tasks.run_recovery()  # what a restarted worker does on startup
    assert summary["chunks_requeued"] == 3
    assert job_service.job_snapshot(job_id)["status"] == "completed"
    assert calls == {0.0: 1, 8.0: 1, 18.0: 1, 28.0: 1}


def test_recovery_requeues_lost_queued_job_and_unfinalized_job(volga_env, tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from workers import transcription_tasks as tasks

    lost = _create_job(tmp_path, 3.0, job_id="lost")  # enqueue never reached Redis
    unfinalized = _create_job(tmp_path, 3.0, job_id="unfinal")
    job_service.claim_job(unfinalized)
    chunk = job_service.ensure_chunks(unfinalized, 3.0)[0]
    with monkeypatch.context() as m:
        m.setattr(tasks, "_enqueue_finalize", lambda job_id: None)  # worker dies before finalizing
        tasks.transcribe_chunk.run(chunk.id)

    with session_scope() as s:
        s.execute(update(TranscriptionJob).where(TranscriptionJob.id == lost)
                  .values(updated_at=utcnow() - timedelta(hours=1)))
    assert job_service.job_snapshot(unfinalized)["status"] == "processing"

    summary = tasks.run_recovery()
    assert summary["jobs_requeued"] == 1 and summary["jobs_finalized"] == 1
    assert job_service.job_snapshot(lost)["status"] == "completed"
    assert job_service.job_snapshot(unfinalized)["status"] == "completed"


def test_cancel_during_processing_stops_new_chunks(volga_env, tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from workers.transcription_tasks import enqueue_job

    job_id = _create_job(tmp_path, 35.0)

    def cancel_after_first(start: float, n: int):  # type: ignore[no-untyped-def]
        if start == 8.0:
            job_service.cancel_job(job_id)
        return None

    calls = _instrument(monkeypatch, cancel_after_first)
    enqueue_job(job_id)
    snap = job_service.job_snapshot(job_id)
    assert snap["status"] == "cancelled"
    assert set(calls) == {0.0, 8.0}  # chunk in flight finished; later chunks never started
    assert snap["chunks"]["completed"] == 2


def test_retention_cleanup(volga_env, tmp_path) -> None:  # type: ignore[no-untyped-def]
    from workers.transcription_tasks import enqueue_job, run_cleanup

    old = _create_job(tmp_path, 3.0, job_id="old")
    fresh = _create_job(tmp_path, 3.0, job_id="fresh")
    enqueue_job(old)
    enqueue_job(fresh)
    storage = get_storage()
    old_job = job_service.get_job(old)
    assert storage.exists(old_job.audio_storage_key)
    assert storage.exists(f"{old_job.transcript_storage_prefix}/transcript.json")

    with session_scope() as s:
        s.execute(update(TranscriptionJob).where(TranscriptionJob.id == old)
                  .values(updated_at=utcnow() - timedelta(days=40)))
    assert run_cleanup(0) == 0  # 0 = keep forever
    assert run_cleanup(30) == 1
    assert job_service.get_job(old) is None and not storage.exists(old_job.audio_storage_key)
    assert job_service.get_job(fresh).status == JobStatus.COMPLETED
