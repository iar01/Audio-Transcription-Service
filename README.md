# Audio Transcription Service

An asynchronous service that turns audio recordings into timestamped transcripts. Upload a file and get back text split into segments, each with a start time, an end time and a confidence score. Transcripts can be downloaded as **TXT**, **SRT** or **JSON**.

The service runs as a web app with a REST API, and also as a command-line tool. Transcription uses [faster-whisper](https://github.com/SYSTRAN/faster-whisper) on the CPU.

**Stack:** FastAPI · Celery · Redis · PostgreSQL / SQLite · FFmpeg · faster-whisper · Docker

---

## Contents

- [Quick start (Docker)](#quick-start-docker)
- [Local setup (without Docker)](#local-setup-without-docker)
- [Configuration](#configuration)
- [Database](#database)
- [Architecture](#architecture)
- [API reference](#api-reference)
- [Project structure](#project-structure)

---

## Quick start (Docker)

**Prerequisites:** [Git](https://git-scm.com/downloads) and [Docker Desktop](https://www.docker.com/products/docker-desktop/) (or Docker Engine with the Compose plugin). Python, FFmpeg, Redis and PostgreSQL all run inside containers; nothing else needs to be installed.

```bash
git clone https://github.com/iar01/Audio-Transcription-Service.git
cd Audio-Transcription-Service
docker compose up --build
```

Open **http://localhost:8000** once the containers are running.

> The first transcription takes longer, because the Whisper model is downloaded on first use and then cached.

This starts four services:

| Service | Role |
|---|---|
| `api` | FastAPI app and web UI on port 8000 |
| `worker` | Celery worker running faster-whisper |
| `redis` | Job queue |
| `postgres` | Database |

Data persists across restarts in Docker volumes (`pgdata`, `redisdata`, `storage`, `models`).

### Common commands

```bash
docker compose up -d --build                        # start in the background
docker compose logs -f worker                       # follow worker logs
docker compose up -d --scale worker=3               # run 3 workers in parallel
docker compose exec api pytest tests/               # run the test suite
docker compose exec api python manage.py cleanup    # delete expired jobs
docker compose down                                 # stop (data is kept)
docker compose down -v                              # stop and delete all data
```

---

## Local setup (without Docker)

**Prerequisites:** Python 3.12, FFmpeg (on your `PATH`) and Redis. On Windows, run Redis through WSL, Memurai or Docker.

```bash
git clone https://github.com/iar01/Audio-Transcription-Service.git
cd Audio-Transcription-Service

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # Windows: copy .env.example .env
```

Start the API and a worker in two terminals:

```bash
uvicorn app:app --port 8000                                        # API + web UI
celery -A workers.celery_app worker --loglevel=INFO --pool=solo    # worker
```

Open **http://127.0.0.1:8000**. To process more jobs at once, start more workers with unique names (`-n worker2@%h`, `-n worker3@%h`, …).

**Command line (no Redis or workers required):**

```bash
python -m transcribe_pipeline.cli transcribe audio.mp3 --out-dir out/
```

**Tests:**

```bash
pytest tests/
```

Tests that need a live Redis are skipped automatically if Redis isn't running.

---

## Configuration

Settings are read from environment variables or a `.env` file in the project root. Copy `.env.example` to `.env` to get started; it documents every option. With Docker, Compose loads `.env` automatically.

| Variable | Default | Purpose |
|---|---|---|
| `REDIS_URL` | `redis://localhost:6379/0` | Celery broker |
| `DATABASE_URL` | *(empty: SQLite)* | Database connection string, e.g. `postgresql+psycopg://...` |
| `WHISPER_MODEL` | `base` | Whisper model size |
| `CELERY_CONCURRENCY` | `1` | Transcriptions per worker |
| `CHUNKING_THRESHOLD` | `600` | Files longer than this many seconds are chunked |
| `MAX_UPLOAD_SIZE_MB` | `500` | Upload size limit |
| `RETENTION_DAYS` | `30` | How long completed jobs are kept |
| `API_PORT` | `8000` | Host port for the API (Docker only) |

**Redis and database URLs in Docker.** Compose always sets these values, so whatever `.env` contains for them is ignored inside the containers:

```
REDIS_URL=redis://redis:6379/0
DATABASE_URL=postgresql+psycopg://transcriber:transcriber@postgres:5432/transcription
```

`redis` and `postgres` are the service names on Compose's private network; inside a container, `localhost` refers to the container itself. Your `.env` can keep `localhost` values for local development. Redis and PostgreSQL aren't exposed on host ports.

> **Security:** the default database password is for local development only. Change it in `docker-compose.yml` before deploying anywhere public.

---

## Database

**Docker.** The database and tables are created automatically; no setup is needed:

1. On first start, the `postgres` container creates the user and database from `POSTGRES_USER`, `POSTGRES_PASSWORD` and `POSTGRES_DB`.
2. When the `api` service starts, it creates any missing tables. Restarts never wipe data. The worker waits for the API to be healthy, so the tables exist before any job runs.

```bash
docker compose exec api python manage.py init-db                                # create tables manually (optional)
docker compose exec postgres psql -U transcriber -d transcription -c "\dt"     # list tables
```

**Local.** With `DATABASE_URL` empty, a SQLite database is created under `storage/` and its tables are created when the API starts (or with `python manage.py init-db`). To use PostgreSQL, create a database and set `DATABASE_URL=postgresql+psycopg://user:password@localhost:5432/transcription`.

**Schema changes.** There are no migrations. Changing a model in `database/models.py` doesn't alter existing tables, and changing the `POSTGRES_*` credentials only takes effect on an empty volume. In either case, reset with `docker compose down -v` (this deletes all data) and run `docker compose up --build`.

---

## Architecture

```mermaid
flowchart LR
    U[Browser / Client] -->|upload| API[FastAPI]
    API -->|save audio| FS[(File storage)]
    API -->|create job| DB[(Database)]
    API -->|queue job| R[Redis]
    R --> W[Celery workers]
    W -->|read audio| FS
    W -->|faster-whisper| W
    W -->|save segments & status| DB
    W -->|TXT / SRT / JSON| FS
    API -->|progress over SSE| U
```

1. **Upload.** The API checks the file type and size, and uses `ffprobe` to confirm the file contains real audio. It saves the file, creates a job and replies with **`202 Accepted`** and a job ID. The API never transcribes anything itself.
2. **Queue.** The job is placed in **Redis**, and **Celery** workers pick jobs up one at a time. Each worker loads the Whisper model once and reuses it for every job.
3. **Normalize.** Every input is converted with FFmpeg to **16 kHz mono WAV**, so the model always receives the same kind of audio.
4. **Chunk long files.** Recordings over 10 minutes are split into 5-minute chunks with 3 seconds of overlap, each processed as its own task. Timestamps are shifted back to their place in the full recording, and every overlap belongs to exactly one chunk, so no words repeat at the seams.
5. **Save incrementally.** Each chunk's segments are written to the database as soon as that chunk finishes. Once every chunk is done, the transcript is assembled and exported.
6. **Track progress.** The browser receives live status through **Server-Sent Events** (SSE), and falls back to polling if SSE isn't available.

### Reliability

- **Retries.** Temporary errors are retried with exponential backoff. Permanent errors, such as corrupt audio, fail immediately.
- **Partial retry.** If a job fails, `POST /retry` reprocesses only the failed chunks.
- **Crash recovery.** If a worker dies, Redis hands its task to another worker. On startup, each worker re-queues stuck chunks and jobs.
- **Idempotency.** The database is the source of truth. Status changes are atomic, and a claim token on each chunk means a task delivered twice can't save its results twice.
- **Cancellation.** Cancelling a job stops it from starting any new chunks.

### Storage

- **Audio files** are saved under a generated name, never the user's filename, at `storage/files/audio/<user>/<job_id>/`.
- **Exported transcripts** are saved at `storage/files/transcripts/<user>/<job_id>/`.
- **The database** holds only jobs, chunks and segments, never audio.
- **Cleanup.** `python manage.py cleanup` deletes jobs older than `RETENTION_DAYS`.

---

## API reference

| Method & path | Description |
|---|---|
| `POST /api/v1/transcriptions` | Upload audio (form field `audio_file`); returns `202` with a `job_id` |
| `GET /api/v1/transcriptions/{id}` | Job status, progress and any error |
| `GET /api/v1/transcriptions/{id}/events` | Live progress stream (SSE) |
| `GET /api/v1/transcriptions/{id}/result` | Transcript text and segments |
| `GET /api/v1/transcriptions/{id}/result/{txt\|srt\|json}` | Download the transcript as a file |
| `POST /api/v1/transcriptions/{id}/retry` | Retry the failed chunks |
| `DELETE /api/v1/transcriptions/{id}` | Cancel the job |
| `GET /api/v1/health` | Database and Redis health |

Supported formats: **WAV, MP3, M4A, FLAC, OGG**.

Errors are returned as `{"error": {"code": "...", "message": "..."}}`:

| Status | Meaning |
|---|---|
| `400` | No file, or the file is empty |
| `409` | The job isn't in the right state for this action |
| `413` | File is too large |
| `415` | Unsupported format |
| `422` | Corrupt audio |
| `503` | The queue is unavailable |

---

## Project structure

```
api/                  REST routes (/api/v1/transcriptions, health) and error format
workers/              Celery app and tasks (transcribe, chunk, finalize, recover, cleanup)
services/             Job state changes, engine loading, error classification
chunking/             Splitting long audio and merging chunk results
transcribe_pipeline/  Validation, FFmpeg normalization, Whisper engine, exporters, CLI
database/             SQLAlchemy models (jobs, chunks, segments) and session
filestore/            File storage (local)
templates/index.html  Web UI
app.py                FastAPI entry point
manage.py             init-db | recover | cleanup | status
Dockerfile            Container image for the API and worker
docker-compose.yml    Full stack: API, worker, Redis, PostgreSQL
```
