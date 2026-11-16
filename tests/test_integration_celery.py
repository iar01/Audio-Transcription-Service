"""End-to-end tests with a real Redis broker and real Celery worker processes.

Skipped automatically when no Redis is available. To run them:
  * have ``redis-server`` on PATH (a throwaway server is started on a free port), or
  * set ``VOLGA_TEST_REDIS_URL=redis://localhost:6379/15`` to use an existing server.
Also requires ffmpeg (chunk extraction).
"""

from __future__ import annotations

import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import pytest

from tests.conftest import HAS_FFMPEG, configure_env, make_wav

ROOT = Path(__file__).resolve().parent.parent
API = "/api/v1/transcriptions"
IS_WINDOWS = sys.platform.startswith("win")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _redis_ping(url: str) -> bool:
    try:
        import redis

        return bool(redis.Redis.from_url(url, socket_connect_timeout=1).ping())
    except Exception:
        return False


@pytest.fixture(scope="module")
def redis_url() -> Iterator[str]:
    if not HAS_FFMPEG:
        pytest.skip("ffmpeg/ffprobe not installed")
    url = os.environ.get("VOLGA_TEST_REDIS_URL")
    if url:
        if not _redis_ping(url):
            pytest.skip(f"VOLGA_TEST_REDIS_URL={url} is not reachable")
        yield url
        return
    binary = shutil.which("redis-server")
    if not binary:
        pytest.skip("Redis not available: install redis-server or set VOLGA_TEST_REDIS_URL")
    port = _free_port()
    proc = subprocess.Popen([binary, "--port", str(port), "--save", "", "--appendonly", "no"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"redis://127.0.0.1:{port}/0"
    for _ in range(50):
        if _redis_ping(url):
            break
        time.sleep(0.1)
    else:
        proc.kill()
        pytest.skip("Could not start redis-server")
    yield url
    proc.terminate()
    proc.wait(timeout=10)


class WorkerProcess:
    def __init__(self, name: str, log_dir: Path) -> None:
        self.name = name
        self.log_path = log_dir / f"worker_{name}.log"
        self.proc: Optional[subprocess.Popen] = None

    def start(self) -> "WorkerProcess":
        kwargs: Dict[str, object] = {}
        if IS_WINDOWS:
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
        else:
            kwargs["start_new_session"] = True
        self._log_fh = open(self.log_path, "ab")
        self.proc = subprocess.Popen([sys.executable, "-m", "tests.celery_test_worker", self.name],
                                     cwd=str(ROOT), env=os.environ.copy(), stdout=self._log_fh,
                                     stderr=subprocess.STDOUT, **kwargs)
        deadline = time.time() + 30
        while time.time() < deadline:
            if b"ready." in self.log_path.read_bytes():
                return self
            if self.proc.poll() is not None:
                raise RuntimeError(f"worker exited early:\n{self.log_path.read_text(errors='replace')}")
            time.sleep(0.2)
        raise RuntimeError(f"worker not ready:\n{self.log_path.read_text(errors='replace')}")

    def kill(self) -> None:
        """Hard-kill the worker (and its prefork children) without any cleanup: a crash."""
        assert self.proc is not None
        if IS_WINDOWS:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(self.proc.pid)], capture_output=True)
        else:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
        self.proc.wait(timeout=10)
        self._log_fh.close()

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.kill()


@pytest.fixture
def live_env(redis_url: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):  # type: ignore[no-untyped-def]
    """Non-eager environment shared by the test process (API) and worker subprocesses."""
    engine_log = tmp_path / "engine.log"
    engine_log.touch()
    workers: List[WorkerProcess] = []

    def configure(**overrides: str) -> None:
        env = {
            "REDIS_URL": redis_url,
            "CELERY_TASK_ALWAYS_EAGER": "false",
            "CELERY_QUEUE": f"volga-test-{uuid.uuid4().hex[:8]}",
            "CELERY_POOL": "solo" if IS_WINDOWS else "prefork",
            "CELERY_CONCURRENCY": "1",
            "CHUNK_STALE_TIMEOUT": "5",
            "QUEUED_STALE_TIMEOUT": "5",
            "TEST_ENGINE_LOG": str(engine_log),
            "TEST_ENGINE_DELAY": "1.5",
        }
        env.update(overrides)
        configure_env(monkeypatch, tmp_path, env)

    def start_worker(name: str) -> WorkerProcess:
        w = WorkerProcess(name, tmp_path).start()
        workers.append(w)
        return w

    def engine_events() -> List[Tuple[int, str, str, float]]:
        out = []
        for line in engine_log.read_text().splitlines():
            pid, event, label, ts = line.split()
            out.append((int(pid), event, label, float(ts)))
        return out

    from fastapi.testclient import TestClient

    import app as app_module

    configure()
    with TestClient(app_module.app) as client:
        yield {"client": client, "configure": configure, "start_worker": start_worker,
               "events": engine_events, "tmp": tmp_path}
    for w in workers:
        w.stop()


def _wait_all(client, job_ids: List[str], status: str, timeout: float) -> Dict[str, dict]:  # type: ignore[no-untyped-def]
    deadline = time.time() + timeout
    snaps: Dict[str, dict] = {}
    while time.time() < deadline:
        snaps = {j: client.get(f"{API}/{j}").json() for j in job_ids}
        if all(s["status"] == status for s in snaps.values()):
            return snaps
        time.sleep(0.25)
    raise AssertionError(f"jobs did not reach {status}: {[(j, s['status']) for j, s in snaps.items()]}")


def _max_overlap(events: List[Tuple[int, str, str, float]]) -> int:
    level = peak = 0
    for _pid, event, _label, _ts in sorted(events, key=lambda e: (e[3], e[1] == "start")):
        level += 1 if event == "start" else -1
        peak = max(peak, level)
    return peak


def test_concurrent_uploads_processed_independently_by_real_workers(live_env) -> None:  # type: ignore[no-untyped-def]
    concurrency = 1 if IS_WINDOWS else 2
    live_env["configure"](CELERY_CONCURRENCY=str(concurrency), TEST_ENGINE_DELAY="1.5")
    live_env["start_worker"]("w1")
    if IS_WINDOWS:  # solo pool: scale out with processes instead
        live_env["start_worker"]("w2")
        concurrency = 2
    client = live_env["client"]

    files = [make_wav(live_env["tmp"] / f"u{i}.wav", 2.0) for i in range(4)]
    results: List[Tuple[int, float, str]] = [(0, 0.0, "")] * len(files)

    def upload(i: int) -> None:
        t0 = time.time()
        with open(files[i], "rb") as fh:
            r = client.post(API, files={"audio_file": (files[i].name, fh, "audio/wav")})
        results[i] = (r.status_code, time.time() - t0, r.json().get("job_id", ""))

    threads = [threading.Thread(target=upload, args=(i,)) for i in range(len(files))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert all(code == 202 for code, _, _ in results)
    assert all(elapsed < 1.5 for _, elapsed, _ in results), "API must not wait for transcription"
    job_ids = [jid for _, _, jid in results]
    assert len(set(job_ids)) == 4

    _wait_all(client, job_ids, "completed", timeout=60)
    events = live_env["events"]()
    assert sum(1 for e in events if e[1] == "end") == 4
    peak = _max_overlap(events)
    assert peak == concurrency, f"expected {concurrency} jobs in parallel, observed {peak}"


def test_worker_crash_mid_job_recovers_without_redoing_completed_chunks(live_env) -> None:  # type: ignore[no-untyped-def]
    from sqlalchemy import func, select

    from database.models import TranscriptionChunk, TranscriptSegment
    from database.session import session_scope

    live_env["configure"](CELERY_CONCURRENCY="1", TEST_ENGINE_DELAY="2.0")
    worker = live_env["start_worker"]("crashy")
    client = live_env["client"]

    src = make_wav(live_env["tmp"] / "long.wav", 35.0)  # 4 chunks of 10s (+2s overlap)
    with open(src, "rb") as fh:
        job_id = client.post(API, files={"audio_file": ("long.wav", fh, "audio/wav")}).json()["job_id"]

    # Wait until chunk 0 is saved and chunk 1 is mid-transcription, then crash the worker.
    deadline = time.time() + 30
    while time.time() < deadline:
        chunks = client.get(f"{API}/{job_id}").json()["chunks"]
        if chunks["completed"] >= 1 and chunks["processing"] >= 1:
            break
        time.sleep(0.1)
    else:
        raise AssertionError("job never reached the crash point")
    worker.kill()
    assert client.get(f"{API}/{job_id}").json()["status"] == "processing"

    time.sleep(6)  # > CHUNK_STALE_TIMEOUT: the orphaned claim is now recoverable
    live_env["start_worker"]("restarted")  # runs startup recovery
    snap = _wait_all(client, [job_id], "completed", timeout=60)[job_id]
    assert snap["chunks"] == {"total": 4, "completed": 4, "failed": 0, "processing": 0, "pending": 0}

    started = [e[2] for e in live_env["events"]() if e[1] == "start"]
    assert started.count("0.0") == 1, f"completed chunk was re-transcribed: {started}"
    assert started.count("8.0") == 2  # interrupted chunk ran again after recovery
    assert started.count("18.0") == 1 and started.count("28.0") == 1  # duplicate deliveries were skipped

    with session_scope() as s:
        for chunk in s.scalars(select(TranscriptionChunk).where(TranscriptionChunk.job_id == job_id)):
            stored = s.scalar(select(func.count()).select_from(TranscriptSegment)
                              .where(TranscriptSegment.chunk_id == chunk.id))
            assert stored == chunk.segment_count  # no duplicate segment rows
    segments = client.get(f"{API}/{job_id}/result").json()["segments"]
    assert [s["start"] for s in segments] == sorted(s["start"] for s in segments)
