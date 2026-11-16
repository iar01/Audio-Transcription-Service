"""FastAPI web application for audio transcription pipeline.

Provides a simple web interface to upload audio files, view transcription
results with timestamps, and download outputs in JSON or SRT format.

Asynchronous API (``/api/v1/transcriptions``): uploads are validated, stored,
recorded in the database and queued to Celery workers via Redis; the request
returns ``202 Accepted`` immediately. The original synchronous endpoint
``POST /api/transcribe`` is kept for backward compatibility (disable it with
``ENABLE_SYNC_TRANSCRIBE=false``).
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from api.errors import ApiError, api_error_handler
from api.routes.health import router as health_router
from api.routes.transcriptions import router as transcriptions_router
from config import get_settings
from database.session import init_db
from transcribe_pipeline.engines.mock_engine import MockTranscriberEngine
from transcribe_pipeline.errors import PipelineError, ValidationError
from transcribe_pipeline.pipeline import TranscriptionPipeline

settings = get_settings()

# Configure logging
logging.basicConfig(level=settings.log_level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("transcribe_web")


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    init_db()  # idempotent schema initialization
    yield


app = FastAPI(title="Audio Transcription Web App", version="2.0.0", lifespan=lifespan)
app.add_exception_handler(ApiError, api_error_handler)
app.include_router(transcriptions_router)
app.include_router(health_router)

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
INDEX_HTML = TEMPLATES_DIR / "index.html"

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}

# Initialize default pipeline engine (try faster-whisper, fallback to mock if uninstalled)
try:
    from transcribe_pipeline.engines.faster_whisper_engine import FasterWhisperEngine

    _default_engine = FasterWhisperEngine(model_size_or_path="base")
    # Test if faster_whisper import succeeds
    import faster_whisper  # type: ignore

    logger.info("Web app using FasterWhisperEngine.")
except Exception:
    logger.info("faster-whisper not installed; Web app using MockTranscriberEngine.")
    _default_engine = MockTranscriberEngine()

pipeline = TranscriptionPipeline(engine=_default_engine)


@app.get("/", response_class=FileResponse)
async def serve_index() -> FileResponse:
    """Serve the single-page HTML interface."""
    if not INDEX_HTML.exists():
        raise HTTPException(status_code=404, detail="index.html not found.")
    return FileResponse(INDEX_HTML)


@app.get("/api/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok", "engine": _default_engine.name}


@app.post("/api/transcribe", deprecated=True)
def transcribe_audio(audio_file: UploadFile = File(...)) -> JSONResponse:
    """Legacy synchronous endpoint: transcribe inside the request and return results.

    Prefer ``POST /api/v1/transcriptions``. Runs in FastAPI's threadpool (``def``,
    not ``async def``) so a long transcription doesn't block the event loop.
    """
    if not settings.enable_sync_transcribe:
        raise HTTPException(status_code=410, detail="Synchronous transcription is disabled; use /api/v1/transcriptions.")
    if not audio_file.filename:
        raise HTTPException(status_code=400, detail="No audio file selected.")

    suffix = Path(audio_file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{suffix}'. Allowed formats: {allowed}",
        )

    # Save to a temporary file
    temp_dir = tempfile.mkdtemp(prefix="transcribe_web_")
    # Never use the client-supplied name as a path component (path traversal).
    temp_path = Path(temp_dir) / f"upload{suffix}"

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)

        file_size = temp_path.stat().st_size
        if file_size == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        logger.info("Web request transcribing '%s' (%d bytes)", audio_file.filename, file_size)

        try:
            # Check if ffprobe/ffmpeg is available for stream check and conversion
            has_ffmpeg = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
            if not has_ffmpeg:
                logger.info("FFmpeg/ffprobe not on PATH; passing audio directly to engine.")

            result = pipeline.run(
                temp_path,
                verify_stream=has_ffmpeg,
                preprocess=has_ffmpeg,
            )

            data = result.to_dict()
            data["filename"] = audio_file.filename
            data["source_file"] = audio_file.filename
            return JSONResponse(content=data)

        except ValidationError as err:
            raise HTTPException(status_code=400, detail=str(err).replace(str(temp_path), audio_file.filename)) from err
        except PipelineError as err:
            logger.exception("Synchronous transcription failed")
            raise HTTPException(status_code=500, detail="Transcription failed.") from err

    finally:
        # Clean up temporary directory and files
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
