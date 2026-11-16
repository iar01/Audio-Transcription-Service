"""REST API, SSE and export tests (Celery in eager mode, mock engine, SQLite in a temp dir)."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

import pytest

from tests.conftest import make_wav, requires_ffmpeg

API = "/api/v1/transcriptions"


def _upload(client, path: Path, name: str | None = None):  # type: ignore[no-untyped-def]
    with open(path, "rb") as fh:
        return client.post(API, files={"audio_file": (name or path.name, fh, "audio/wav")})


def _wait_status(client, job_id: str, statuses: set, timeout: float = 20.0) -> Dict[str, Any]:  # type: ignore[no-untyped-def]
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"{API}/{job_id}").json()
        if body["status"] in statuses:
            return body
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} never reached {statuses}")


def test_short_audio_transcription(client, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    resp = _upload(client, make_wav(tmp_path / "short.wav", 3.0))
    assert resp.status_code == 202
    body = resp.json()
    assert body["status"] == "queued" and body["message"] == "Transcription queued successfully"
    job = _wait_status(client, body["job_id"], {"completed"})
    assert job["progress"] == 100.0 and job["chunked"] is False and job["chunks"]["total"] == 1
    assert job["started_at"] and job["completed_at"]

    result = client.get(f"{API}/{body['job_id']}/result").json()
    assert result["filename"] == "short.wav"
    assert result["engine"] == "mock"
    assert result["segments"] and result["text"]
    assert "storage" not in json.dumps(result)  # no internal paths leaked


@requires_ffmpeg
def test_long_audio_is_chunked_with_ordered_timestamps(client, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    resp = _upload(client, make_wav(tmp_path / "long.wav", 35.0))
    job = _wait_status(client, resp.json()["job_id"], {"completed"})
    assert job["chunked"] is True and job["chunks"]["total"] == 4 and job["chunks"]["completed"] == 4
    segments = client.get(f"{API}/{job['job_id']}/result").json()["segments"]
    starts = [s["start"] for s in segments]
    assert starts == sorted(starts)
    assert all(0 <= s["start"] <= s["end"] <= 35.0 for s in segments)
    for a, b in zip(segments, segments[1:]):  # no overlapping duplicates at boundaries
        assert b["start"] >= a["start"]
        assert not (b["text"] == a["text"] and b["start"] < a["end"])


def test_rejects_unsupported_extension(client, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    p = tmp_path / "notes.txt"
    p.write_text("hello")
    resp = _upload(client, p)
    assert resp.status_code == 415
    assert resp.json()["error"]["code"] == "unsupported_format"


def test_rejects_empty_file(client, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    p = tmp_path / "empty.wav"
    p.write_bytes(b"")
    resp = _upload(client, p)
    assert resp.status_code == 400 and resp.json()["error"]["code"] == "empty_file"


def test_rejects_oversized_upload(volga_env, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    volga_env(MAX_UPLOAD_SIZE_MB="1")
    from fastapi.testclient import TestClient

    import app as app_module

    with TestClient(app_module.app) as c:
        resp = _upload(c, make_wav(tmp_path / "big.wav", 40.0))  # ~1.3 MB
    assert resp.status_code == 413 and resp.json()["error"]["code"] == "file_too_large"


@requires_ffmpeg
def test_rejects_corrupt_audio(client, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    p = tmp_path / "corrupt.wav"
    p.write_bytes(b"this is definitely not audio" * 100)
    resp = _upload(client, p)
    assert resp.status_code == 422
    assert str(tmp_path) not in resp.text  # no filesystem paths in the error


def test_unknown_job_404_and_result_before_completion_409(client, tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    assert client.get(f"{API}/doesnotexist").status_code == 404
    import workers.transcription_tasks as tasks

    monkeypatch.setattr(tasks, "_publish", lambda task, args: "fake-task-id")  # job stays queued
    job_id = _upload(client, make_wav(tmp_path / "a.wav", 2.0)).json()["job_id"]
    resp = client.get(f"{API}/{job_id}/result")
    assert resp.status_code == 409 and resp.json()["status"] == "queued"
    progress = client.get(f"{API}/{job_id}/progress").json()
    assert progress["status"] == "queued" and progress["progress"] == 0


def test_exports_match_existing_exporters(client, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    from transcribe_pipeline.exporters import format_srt_timestamp

    job_id = _upload(client, make_wav(tmp_path / "e.wav", 3.0)).json()["job_id"]
    _wait_status(client, job_id, {"completed"})
    result = client.get(f"{API}/{job_id}/result").json()

    txt = client.get(f"{API}/{job_id}/result/txt")
    assert txt.status_code == 200 and txt.text.strip() == result["text"]
    assert 'filename="e.txt"' in txt.headers["content-disposition"]

    srt = client.get(f"{API}/{job_id}/result/srt").text
    first = result["segments"][0]
    assert srt.startswith("1\n") and f"{format_srt_timestamp(first['start'])} --> {format_srt_timestamp(first['end'])}" in srt

    data = client.get(f"{API}/{job_id}/result/json").json()
    assert data["segments"] == result["segments"] and data["engine"] == "mock"

    assert client.get(f"{API}/{job_id}/result/docx").status_code == 400


def test_multiple_concurrent_uploads(client, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    files = [make_wav(tmp_path / f"c{i}.wav", 2.0 + i) for i in range(6)]
    responses: List[Any] = [None] * len(files)

    def worker(i: int) -> None:
        responses[i] = _upload(client, files[i])

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(len(files))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert all(r.status_code == 202 for r in responses)
    ids = {r.json()["job_id"] for r in responses}
    assert len(ids) == len(files)
    for job_id in ids:
        assert _wait_status(client, job_id, {"completed"})["status"] == "completed"


def test_queue_unavailable_returns_503_and_job_can_be_retried(client, tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import workers.transcription_tasks as tasks

    real_publish = tasks._publish

    def broken(task, args):  # type: ignore[no-untyped-def]
        raise tasks.QueueUnavailableError("Connection refused")

    monkeypatch.setattr(tasks, "_publish", broken)
    resp = _upload(client, make_wav(tmp_path / "r.wav", 2.0))
    assert resp.status_code == 503
    body = resp.json()
    assert body["error"]["code"] == "queue_unavailable" and "Connection refused" not in resp.text
    job_id = body["job_id"]
    assert client.get(f"{API}/{job_id}").json()["status"] == "failed"

    monkeypatch.setattr(tasks, "_publish", real_publish)  # Redis is back
    assert client.post(f"{API}/{job_id}/retry").status_code == 202
    assert _wait_status(client, job_id, {"completed"})["retry_count"] == 1


def test_cancel_queued_job_and_conflicts(client, tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import workers.transcription_tasks as tasks

    monkeypatch.setattr(tasks, "_publish", lambda task, args: "fake-task-id")
    job_id = _upload(client, make_wav(tmp_path / "x.wav", 2.0)).json()["job_id"]
    resp = client.delete(f"{API}/{job_id}")
    assert resp.status_code == 200 and resp.json()["status"] == "cancelled"
    assert client.delete(f"{API}/{job_id}").status_code == 409

    # A late-delivered task for a cancelled job does nothing
    assert tasks.transcribe_job.run(job_id)["skipped"] is True
    assert client.get(f"{API}/{job_id}").json()["status"] == "cancelled"


def test_retry_rejected_for_completed_job(client, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    job_id = _upload(client, make_wav(tmp_path / "ok.wav", 2.0)).json()["job_id"]
    _wait_status(client, job_id, {"completed"})
    assert client.post(f"{API}/{job_id}/retry").status_code == 409
    assert client.delete(f"{API}/{job_id}").status_code == 409


def _read_sse(client, job_id: str, timeout: float = 15.0) -> List[Dict[str, Any]]:  # type: ignore[no-untyped-def]
    events: List[Dict[str, Any]] = []
    current: Dict[str, Any] = {}
    deadline = time.time() + timeout
    with client.stream("GET", f"{API}/{job_id}/events") as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        for line in resp.iter_lines():
            if line.startswith("event:"):
                current = {"event": line.split(":", 1)[1].strip()}
            elif line.startswith("data:") and current:
                current["data"] = json.loads(line.split(":", 1)[1])
                events.append(current)
                if current["event"] in {"completed", "failed", "cancelled"}:
                    break
                current = {}
            if time.time() > deadline:
                break
    return events


def test_sse_streams_status_progress_and_completion(client, tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import workers.transcription_tasks as tasks

    real_publish = tasks._publish
    monkeypatch.setattr(tasks, "_publish", lambda task, args: "deferred")
    job_id = _upload(client, make_wav(tmp_path / "sse.wav", 35.0)).json()["job_id"]
    monkeypatch.setattr(tasks, "_publish", real_publish)

    def run_later() -> None:
        time.sleep(0.5)
        tasks.transcribe_job.apply(args=[job_id])

    threading.Thread(target=run_later, daemon=True).start()
    events = _read_sse(client, job_id)
    names = [e["event"] for e in events]
    assert names[0] == "status" and events[0]["data"]["status"] == "queued"
    assert names[-1] == "completed" and events[-1]["data"]["progress"] == 100.0
    assert any(e["data"]["status"] == "processing" for e in events)


def test_sse_for_failed_job_and_unknown_job(client, tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    assert client.get(f"{API}/nope/events").status_code == 404
    import workers.transcription_tasks as tasks

    monkeypatch.setattr(tasks, "_publish", lambda task, args: (_ for _ in ()).throw(tasks.QueueUnavailableError("x")))
    job_id = _upload(client, make_wav(tmp_path / "f.wav", 2.0)).json()["job_id"]
    events = _read_sse(client, job_id)
    assert events[-1]["event"] == "failed" and events[-1]["data"]["can_retry"] is True


def test_legacy_sync_endpoint_still_works(client, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    with open(make_wav(tmp_path / "legacy.wav", 2.0), "rb") as fh:
        resp = client.post("/api/transcribe", files={"audio_file": ("legacy.wav", fh, "audio/wav")})
    assert resp.status_code == 200
    data = resp.json()
    assert data["filename"] == "legacy.wav" and data["segments"]


def test_index_page_served(client) -> None:  # type: ignore[no-untyped-def]
    resp = client.get("/")
    assert resp.status_code == 200 and "/api/v1/transcriptions" in resp.text and "bootstrap" in resp.text
