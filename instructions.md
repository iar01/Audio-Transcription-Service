# Task: Build a simple audio transcription pipeline (take-home)

## Context
Take-home assignment. Brief: build a simple pipeline that converts audio into
text and processes the result for downstream use. Any open-source STT library
or API, any language, mock data allowed. The focus is engineering decisions,
not training a model. Reviewers value clear reasoning, so keep the solution
small, modular, and well explained.

The submission form asks these questions, so the code and README must answer
each one:
1. Accepts an audio file (WAV or MP3)
2. Transcribes speech into text
3. Returns the transcription with timestamps per segment
4. How are different audio formats handled?
5. How are long audio files handled?

Part 2 of the form is system design (concurrent uploads, storing audio and
transcripts, retry/recovery, exposing an API). It is WRITING ONLY. Do not
implement any of it. It is already written up in the README.

The submission also needs: a Git repo with a clean history, a README, and a
screen-recorded walkthrough where the author shows the code and a live run.
So the project must be easy to demo.

## Existing README
A draft README.md is already in the repo root. It was written before the code.
Keep its structure and the Part 2 design section unchanged. Your job:
- Make the code match what the README claims. If you deviate from the spec,
  fix the README instead of leaving a false claim.
- Replace the placeholder "Example output" with REAL output from running the
  pipeline on the sample clip.
- Tick the status checkboxes honestly (Phase 1 / Phase 2).
- Keep the glossary.
Do not add claims about features that are not implemented.

## Tech constraints
- Python 3.10+
- STT: faster-whisper, model `base` (configurable), int8, CPU, VAD filter on
- ffmpeg/ffprobe via subprocess; fail with a clear error if missing
- CLI via argparse
- Tests via pytest
- Minimal dependencies, pinned in requirements.txt
- No web UI, Docker, queues, databases, or cloud deployment, and no API server
  (the API is design-only)

## Layout
```
transcribe_pipeline/
  models.py          # Segment dataclass
  errors.py          # custom exceptions
  validate.py
  preprocess.py
  engines/
    base.py          # TranscriberEngine Protocol: transcribe(path) -> list[Segment]
    faster_whisper_engine.py
    mock_engine.py   # canned segments, no model download
  exporters.py
  pipeline.py
  cli.py
tests/
  data/              # short sample clip + known transcript
README.md
requirements.txt
.gitignore
```

## Phase 1: Core pipeline (must be complete and tested first)

**Segment**: start (float seconds), end (float seconds), text (str),
confidence (float | None).

**Validate** (form Q1): file exists, non-empty, under a configurable size
limit, extension in {wav, mp3, m4a, flac, ogg}. Confirm the real content with
ffprobe rather than trusting the extension. Raise specific exceptions for
missing, empty, unsupported, or corrupt files.

**Preprocess** (form Q4): convert every input to 16 kHz mono WAV with ffmpeg
into a temp directory, so the engine always receives one consistent format.
Detect missing ffmpeg. Clean up temp files even on failure.

**Transcribe** (form Q2 and Q3): call the engine through the interface.
faster-whisper returns segments as a generator, so consume it lazily and
build Segment objects with start, end, text, and confidence = exp(avg_logprob).
A silent file yields zero segments, which is a valid empty result with a
logged warning.

**Long audio** (form Q5): rely on faster-whisper's internal 30-second
windowing and lazy segment generator so memory stays flat, plus the VAD
filter to skip silence. Log progress as segments arrive. Do NOT build
chunking. It is documented in the README as a next step.

**Export**: write JSON (segments plus metadata: source file, engine, model,
language, duration), SRT (correct HH:MM:SS,mmm formatting, sequential
indices), and plain TXT into an output directory.

**CLI**:
```
python -m transcribe_pipeline.cli transcribe <audio> --out-dir out/ \
    [--engine faster-whisper|mock] [--model base] [--verbose]
```
Exit 0 on success. On failure, print a one-line readable error and exit
non-zero (traceback only with --verbose). Use the `logging` module with
per-stage timing. Logs should be readable on screen during a recorded demo.

**Tests (Phase 1)**:
- validate: missing, empty, wrong type, corrupt file
- exporters: SRT timestamp edge cases (0.0, 59.999, 3661.5), JSON structure
- end-to-end with the mock engine (no model download, no network)
- one short sample clip in tests/data (generate with espeak or pyttsx3, or use
  a public-domain recording) with its known transcript, plus an optional test
  skipped when faster-whisper isn't installed that runs the real engine and
  asserts a rough word error rate under a loose threshold

Run pytest after each module. Do not start Phase 2 until Phase 1 passes.

## Phase 2: Optional extras (only if Phase 1 is done and passing)
- `postprocess.py`: normalize whitespace and punctuation spacing, capitalize
  sentence starts, extract top-N keywords with YAKE (rule-based, no LLM)
- `search.py` plus a CLI `search <json> "term"` command: case-insensitive,
  supports multi-word phrases, returns timestamp and a short snippet
- Unit tests for both
If skipped, leave the Phase 2 checkbox unticked in the README.

## Git workflow
- Initialize a repo, add a `.gitignore` (venv, __pycache__, model caches,
  temp files, out/).
- Make small commits with meaningful messages, in build order, for example:
  "Add Segment model and custom errors", "Add input validation with ffprobe
  check", "Add ffmpeg preprocessing", "Add mock engine and exporters", and so
  on. No single giant commit.
- Run pytest before each commit that touches code.

## Demo readiness
- A fresh clone must work with: create venv, `pip install -r requirements.txt`,
  run the CLI on the sample clip, and see the JSON/SRT/TXT in out/.
- Commit the sample clip so the demo needs no external files.
- Give the files short, clear module docstrings so the code reads well when
  scrolled through on screen.

## Working rules
- Build order: models and errors, validate, preprocess, mock engine, exporters,
  pipeline, CLI, real engine, tests, README fixes, then Phase 2 if time allows.
- Type hints and short docstrings on public functions. No dead code.
- Don't add features beyond this spec. If something is ambiguous, choose the
  simplest option and record it in the README.
- When finished, report: what was built, what was tested, which README
  statements you changed, and any deviations from this spec.