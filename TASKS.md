# VoiceForge — Tasks

Local-first desktop app for AI voice cloning. Built around XTTS v2, FastAPI, and Gradio (MVP) → Tauri (later).

**Guiding principle:** UX is the product. No ML jargon ever leaks to the user. "Set up your voice profile" — not "train a model."

---

## M0 — Foundation (60% done)

Goal: backend skeleton runs, XTTS inference works end-to-end via API.

- [x] XTTS v2 loads and generates audio (`test_xtts.py`)
- [x] FastAPI app scaffolded (`main.py`, inference router, job manager)
- [x] `inference_service.py` working (voice cloning + built-in speaker fallback)
- [x] Hardware detection (`backend/system/hardware.py`)
- [x] **M0 gate:** `uvicorn backend.main:app` boots cleanly
- [x] **M0 gate:** `GET /health` returns 200
- [x] **M0 gate:** `GET /system` returns valid GPU/RAM/disk profile
- [x] **M0 gate:** `POST /tts` generates a valid `.wav` end-to-end
- [x] Fix CUDA/torchaudio mismatch (libcudart.so.13 issue) — picked CPU build (torch==2.5.1+cpu, torchaudio==2.5.1+cpu)
- [x] Wire `low_vram_mode` from `hardware.py` into a runtime config flag
- [ ] Add `pytest` smoke tests for `/health`, `/system`, `/tts`
- [x] Update `.gitignore` (pycache, data/cache, data/temp, models, .env)

---

## M1 — Recording + Preprocessing + MVP UI

Goal: user can record 30 clips through a guided UI, app validates and cleans audio.

### Backend — `backend/audio/`

- [ ] `recorder.py` — accept uploaded `.wav` chunks, save to `data/projects/{project}/raw/`
- [ ] `validator.py` — per-clip checks:
  - [ ] Duration in range (e.g. 3–15s)
  - [ ] SNR / silence ratio threshold
  - [ ] Clipping detection (peak > -1 dBFS)
  - [ ] Sample rate normalization target (24 kHz for XTTS)
  - [ ] Return user-friendly error messages (no jargon)
- [ ] `preprocessor.py` — FFmpeg pipeline:
  - [ ] Resample to 24 kHz mono
  - [ ] Trim leading/trailing silence
  - [ ] Loudness normalize (EBU R128 or peak-based)
  - [ ] Optional: light denoise pass
  - [ ] Output to `data/projects/{project}/processed/`
- [ ] `transcriber.py` — faster-whisper integration for clip QA (verify spoken text matches script)
- [ ] Recording script asset (`data/scripts/default_en.json`) — ~30 phonetically diverse prompts

### API

- [ ] `POST /projects` — create a new voice project
- [ ] `GET /projects/{id}` — project state, clip count, validation status
- [ ] `POST /projects/{id}/clips` — upload a clip (multipart)
- [ ] `DELETE /projects/{id}/clips/{clip_id}` — re-record support
- [ ] `POST /projects/{id}/preprocess` — kick off async preprocess job
- [ ] Wire all long-running work through `JobManager` (background tasks)

### Frontend — Gradio MVP

- [ ] App shell with project picker + create-new flow
- [ ] Hardware check screen on first launch ("Your machine is ready" / "Compatibility notice")
- [ ] Recording flow:
  - [ ] Show prompt text, record button, waveform preview
  - [ ] Inline validation feedback ("Try again — too quiet" / "Looks great")
  - [ ] Progress: clip X of 30
- [ ] Review screen — list clips, replay, re-record
- [ ] "Set up your voice profile" CTA → triggers preprocess + train
- [ ] Status panel — friendly progress messages, no percentages of loss

### Persistence

- [ ] SQLite schema via SQLAlchemy: `projects`, `clips`, `jobs`
- [ ] Migration / init on first run

---

## M2 — Training Pipeline (XTTS Fine-Tune)

Goal: locally fine-tune XTTS v2 on the user's clips, with low-VRAM defaults that just work on a 4 GB GTX 1650.

### Pipeline — `backend/pipelines/training.py`

- [ ] Dataset builder — convert processed clips + transcripts into XTTS training format
- [ ] Auto-config based on `hardware.py`:
  - [ ] GPU + ≥8 GB VRAM → standard config
  - [ ] GPU + <8 GB VRAM → low-VRAM (fp16, gradient checkpointing, batch_size=2, grad accumulation)
  - [ ] CPU only → warn + offer cloud fallback or refuse with friendly message
- [ ] Training loop wrapper with progress callbacks → `JobManager.update_progress`
- [ ] Checkpoint saving to `data/projects/{id}/checkpoints/`
- [ ] Early-stopping / best-checkpoint selection
- [ ] Validation sample synthesis after each epoch (for UI preview)

### API

- [ ] `POST /projects/{id}/train` — start training job
- [ ] `GET /jobs/{id}` — poll status (already scaffolded, wire it up)
- [ ] `POST /jobs/{id}/cancel` — clean cancellation

### UX

- [ ] Single-button "Set up voice profile" — no training params exposed
- [ ] Friendly progress copy: "Listening to your voice...", "Almost ready..."
- [ ] ETA estimation based on hardware profile
- [ ] Preview clip auto-plays when training reaches a usable checkpoint
- [ ] Background-safe: user can close UI, training continues, status restored on relaunch

---

## M3 — Inference with Cloned Voice + Export

Goal: user generates speech with their voice and exports the trained model.

- [ ] `POST /projects/{id}/synthesize` — TTS with the project's fine-tuned checkpoint
- [ ] UI: text box → generate → playback + download
- [ ] Multi-language support (XTTS supports 17+) — language picker in UI
- [ ] Speed/emphasis controls (kept simple — sliders, not params)
- [ ] Export flow:
  - [ ] `POST /projects/{id}/export` — package checkpoint + metadata + license note as `.zip`
  - [ ] UI: "Save voice profile" button → file dialog
  - [ ] Import flow: drop a `.zip` to restore a project

---

## M4 — Desktop Packaging (Tauri)

Goal: ship as a real desktop app, not a Python script.

- [ ] Decide bundling strategy: Tauri shell + sidecar Python backend
- [ ] Package Python runtime + deps (PyInstaller or shiv) as sidecar binary
- [ ] Bundle FFmpeg statically
- [ ] First-run flow: model download with progress + resumability
- [ ] Auto-update channel (optional)
- [ ] Code-signing for macOS / Windows (when ready)
- [ ] Replace Gradio UI with Tauri + web frontend (React/Svelte/etc.)

---

## Cross-cutting

### Hardware + Performance

- [ ] Document tested configs (GTX 1650 4GB, M3 Pro CPU) in README
- [ ] Memory guard: refuse training if free VRAM < threshold, suggest cloud option
- [ ] Crash-safe checkpointing — resume from last good state

### Observability

- [ ] Structured logs already in place — add request IDs to API logs
- [ ] Optional anonymous telemetry (opt-in, off by default)
- [ ] User-facing error reporter ("Send report" button → bundles last log + system profile)

### Quality

- [ ] Test suite: unit (audio validator, preprocessor) + integration (API smoke tests)
- [ ] CI: lint + test on push (later)
- [ ] README: install, run, troubleshoot (CUDA, FFmpeg)

### UX Copy Audit

- [ ] Replace any leaked jargon: "epoch" → "round", "loss" → hidden, "checkpoint" → "saved progress"
- [ ] Error messages always action-oriented ("Move closer to the mic" not "SNR below threshold")
- [ ] First-run onboarding — explain what to expect in plain language

---

## Open Questions

- [ ] CPU-only training on M3 Pro — feasible for full fine-tune, or restrict to inference + voice-cloning-via-reference only?
- [ ] Licensing terms for exported voice models (Coqui XTTS license implications)
- [ ] Recording script localization beyond English
