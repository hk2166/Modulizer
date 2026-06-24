# VoiceForge — Tasks

Local-first desktop app for AI voice cloning. Built around XTTS v2, FastAPI, and Gradio (MVP) → Tauri (later).

**Guiding principle:** UX is the product. No ML jargon ever leaks to the user. "Set up your voice profile" — not "train a model."

---

## Priority legend

Every task is tagged with a priority weighted by the product strategy (Quick Clone is the primary path; persistence + tests are the notable open gaps; M4 is required to ship but comes later).

- **(P0)** — Critical / blocking. A milestone gate or something everything else depends on.
- **(P1)** — High. Core to the milestone it lives in or to shipping.
- **(P2)** — Medium. Important but not blocking; polish or robustness.
- **(P3)** — Low. Nice-to-have or explicitly post-MVP.

**Top open priorities right now:** pytest smoke tests (P1), SQLite persistence + job state (P1), and the M4 packaging spine — random-port sidecar, bundling, model-download-with-progress, Tauri frontend (P1).

---

## Product Strategy

VoiceForge ships **two voice cloning paths** with very different hardware demands:

### Quick Clone (default, ships first) — `reference cloning`
- User records **1 clip** (6–10 seconds).
- Generates speech in their voice **immediately** (a few seconds per sentence on CPU).
- Works on **any** machine — no GPU required, no training time, no big downloads after initial model fetch.
- **This is the primary product path** every user goes through on first run.

### Voice Profile (advanced, gated) — `fine-tuning`
- User records **~30 clips** (3–15s each), then trains a custom model.
- Higher quality cloning, can carry style/emphasis better.
- **Requires:** GPU with ≥4 GB VRAM, ~2–5 GB free disk, **2–6 hours of training time**.
- CPU-only fine-tuning is technically possible but takes 24+ hours — we **refuse it** with a friendly explanation rather than ship a broken experience.
- UI surfaces resource cost up front: "This will take ~3 hours and use your GPU. Continue?"

---

## M0 — Foundation

Goal: backend skeleton runs, XTTS inference works end-to-end via API.

- [x] **(P0)** XTTS v2 loads and generates audio (`test_xtts.py`)
- [x] **(P0)** FastAPI app scaffolded (`main.py`, inference router, job manager)
- [x] **(P0)** `inference_service.py` working (voice cloning + built-in speaker fallback)
- [x] **(P1)** Hardware detection (`backend/system/hardware.py`)
- [x] **(P0)** **M0 gate:** `uvicorn backend.main:app` boots cleanly
- [x] **(P0)** **M0 gate:** `GET /health` returns 200
- [x] **(P1)** **M0 gate:** `GET /system` returns valid GPU/RAM/disk profile
- [x] **(P0)** **M0 gate:** `POST /tts` generates a valid `.wav` end-to-end
- [x] **(P0)** Fix CUDA/torchaudio mismatch (libcudart.so.13) — picked CPU build
- [x] **(P1)** Wire `low_vram_mode` from `hardware.py` into a runtime config flag
- [x] **(P2)** Update `.gitignore` (pycache, data/cache, data/temp, models, .env)
- [ ] **(P1)** Add `pytest` smoke tests for `/health`, `/system`, `/tts`
- [x] **(P1)** **User data directory helper** — `get_user_data_dir()` returns `%APPDATA%/VoiceForge` on Windows, `~/Library/Application Support/VoiceForge` on macOS, `~/.local/share/voiceforge` on Linux. Replace all uses of `DATA_DIR` with this. Cheap now, painful later.
- [x] **(P2)** Regenerate `requirements-lock.txt` from a clean venv (current lock has ROS 2 packages from dev machine)

---

## M1 — Quick Voice Clone (Primary product path)

Goal: user installs the app, records one clip, types text, hears their cloned voice. Works on any hardware. **This is the default and only path most users will use.**

### Backend — `backend/audio/`

- [x] **(P0)** `recorder.py` — accept uploaded `.wav` chunks, save to `data/projects/{project}/raw/`
- [x] **(P0)** `validator.py` — per-clip checks:
  - [x] **(P1)** Duration in range (3–15s)
  - [x] **(P1)** SNR / silence ratio threshold
  - [x] **(P1)** Clipping detection (peak > -1 dBFS)
  - [x] **(P1)** Sample rate normalization target (24 kHz)
  - [x] **(P1)** User-friendly error messages (no jargon)
- [x] **(P0)** `preprocessor.py` — librosa pipeline:
  - [x] **(P1)** Resample to 24 kHz mono
  - [x] **(P1)** Trim leading/trailing silence
  - [x] **(P1)** Peak normalize loudness
  - [x] **(P2)** Optional spectral denoise pass
  - [x] **(P1)** Output to `data/projects/{project}/processed/`
- [x] **(P1)** `transcriber.py` — faster-whisper for clip QA (verify spoken text matches script)
- [x] **(P2)** Recording prompts asset (`data/scripts/default_prompts.json`) — 30 EN + 30 HI phonetically diverse prompts
- [x] **(P2)** `cleaner.py` — synthesis-targeted reference cleaning (DC offset, high-pass @ 80 Hz, RMS normalize, soft limiter). Off by default — XTTS clones better from natural processed audio.
- [x] **(P1)** `importer.py` — long audio/video import via bundled ffmpeg; concatenates speech bursts and chunks into uniform ~10s clips (handles both continuous and spotty source material).

### API — Project lifecycle

- [x] **(P0)** `POST /projects` — create a new voice project
- [x] **(P0)** `GET /projects/{id}` — project state, clip count, validation status
- [x] **(P0)** `POST /projects/{id}/clips` — upload a clip (multipart)
- [x] **(P1)** `GET /projects/{id}/clips` — list all clips for a project
- [x] **(P1)** `DELETE /projects/{id}/clips/{clip_id}` — re-record support
- [x] **(P0)** `POST /projects/{id}/preprocess` — kick off async preprocess job
- [x] **(P0)** `GET /jobs/{job_id}` — poll job status / progress / result
- [x] **(P0)** All long-running work wired through `JobManager` (background tasks)
- [x] **(P1)** `POST /projects/{id}/import` — multipart upload of long audio/video; segments and saves as project clips (background job)

### API — Reference cloning (synth)

- [x] **(P0)** `POST /projects/{id}/synthesize` — text + reference clip → generated `.wav`
  - Picks the first valid processed clip as the reference automatically (user shouldn't choose one)
  - Accepts `text` and `language`
  - Returns generated audio path
- [x] **(P1)** `GET /projects/{id}/preview/{clip_id}` — stream a generated `.wav` back to the UI

### Frontend — Gradio MVP

- [x] **(P0)** App shell with project picker + create-new flow
- [x] **(P1)** **First-run hardware check** — "Your machine is ready" / "Compatibility notice for fine-tuning" (Quick Clone always available)
- [x] **(P0)** **Quick Clone flow** (the main UX):
  - [x] **(P0)** Show one prompt, record button, waveform preview
  - [x] **(P1)** Inline validation feedback ("Try again — too quiet" / "Looks great")
  - [x] **(P1)** Auto-preprocess on validation pass
  - [x] **(P0)** Text input → generate → playback
- [x] **(P1)** Status panel — friendly progress messages, no percentages of loss
- [x] **(P2)** **Long audio / video import** — drop a podcast, interview, or video; auto-segment into 10s clips usable as references (handles spotty speech via concatenate-then-chunk)

### Persistence

- [ ] **(P1)** SQLite schema via SQLAlchemy: `projects`, `clips`, `jobs`
- [ ] **(P1)** Migration / init on first run
- [ ] **(P1)** Job state persisted to SQLite (so closing the app doesn't lose status)

---

## M2 — Voice Profile (Advanced, fine-tuning)

Goal: power users with capable hardware can fine-tune XTTS on 30 clips for higher fidelity. **Strictly opt-in, hardware-gated, with up-front time and resource warnings.**

### UX — Resource gating

- [x] **(P1)** **Pre-flight resource check** before kickoff:
  - [x] **(P1)** Free VRAM ≥ 3 GB → allow with low-VRAM config
  - [x] **(P1)** Free VRAM ≥ 8 GB → allow standard config
  - [x] **(P1)** No GPU → refuse with: "Voice Profile training needs a graphics card. You can use Quick Clone instead — it works without one."
  - [x] **(P1)** Free disk < 5 GB → refuse with disk-cleanup suggestion
- [x] **(P1)** **Pre-training disclosure modal** showing:
  - [x] **(P1)** Estimated time based on hardware (e.g. "About 3 hours on your GTX 1650")
  - [x] **(P2)** Estimated power use ("Your GPU will run at full load")
  - [x] **(P2)** What gets saved and where
  - [x] **(P1)** "I understand — start recording →" button
- [ ] **(P2)** Background-safe: user can close Tauri window, training continues, status restored on relaunch (depends on M4 packaging + SQLite job persistence)
- [x] **(P2)** ETA estimation based on hardware profile, updated each round

### Pipeline — `backend/pipelines/training.py`

- [x] **(P1)** Dataset builder — convert processed clips + transcripts into XTTS training format
- [x] **(P1)** Auto-config based on `hardware.py`:
  - [x] **(P1)** GPU + ≥8 GB VRAM → standard config
  - [x] **(P1)** GPU + 3–8 GB VRAM → low-VRAM (fp16, gradient checkpointing, batch_size=2)
  - [x] **(P1)** CPU only → refuse with friendly message (no cloud fallback in v1)
- [x] **(P1)** Training loop wrapper with progress callbacks → `JobManager.update_progress`
  - **Quality follow-ups discovered during this work:**
  - [x] **(P2)** Switch `preprocessor.TARGET_SAMPLE_RATE` from 24000 → 22050 to align with XTTS GPT/dvae native rate. Output stays 24 kHz via the HiFi-GAN decoder. Removes a per-step torchaudio resample.
  - [x] **(P2)** Make `dataset_builder.py` pick a "golden reference" clip (5–7s, near-median pitch + RMS) and store its id in `manifest.json`; teach `project_service.get_reference_clip` to prefer that over `sorted(...)[0]`.
  - [x] **(P2)** Audit `preprocessor._spectral_denoise` — the path is off by default but spectral subtraction strips formants when used; consider replacing with a no-op or a learned model, not a hand-rolled gate.
  - [x] **(P2)** **Hindi (and other Indic-language) pronunciation gap:** XTTS's `tokenizer.preprocess_text` falls through to `basic_cleaners` (lowercase + whitespace) for Hindi — the maintainer left a `# @manmay will implement this` TODO. Same fallback hits Bengali, Tamil, Telugu, Marathi, Gujarati, Punjabi, Urdu. Two ways to close the gap, pick one or do both:
  - [x] **(P2)** **App-layer pre-normalizer (preferred):** add a `backend/audio/text_cleaners.py` with a `hindi_cleaners(text)` that handles schwa-deletion, halant rules, number expansion, and Latin→Devanagari for inevitable English-leak words. Run it inside `inference_service.generate_speech` and `dataset_builder._sanitize_text` before either path hits the model. Libraries to evaluate: `indic-transliteration`, `aksharamukha`. Survives pip upgrades and stays under our control.
  - [ ] **(P3)** **Venv monkey-patch (fallback):** at app startup, replace `TTS.tts.layers.xtts.tokenizer.VoiceBpeTokenizer.preprocess_text` with a wrapper that routes `lang="hi"` through our `hindi_cleaners` before the base call. Brittle (won't survive `pip install -U TTS`), but catches paths inside Coqui that bypass our service layer (training dataset loader, inference_stream, etc.). Only worth doing if option A leaves measurable gaps.
- [x] **(P1)** Cooperative cancellation — training checks a flag every N batches
- [x] **(P1)** Checkpoint saving to `data/projects/{id}/checkpoints/`
- [x] **(P2)** Early-stopping / best-checkpoint selection
- [x] **(P2)** Validation sample synthesis after each round (for UI preview)
- [x] **(P2)** Crash-safe checkpointing — resume from last good state

### API

- [x] **(P1)** `POST /projects/{id}/train` — start training job (returns job_id)
- [x] **(P1)** `POST /jobs/{id}/cancel` — clean cooperative cancellation
- [x] **(P1)** `POST /projects/{id}/synthesize?profile=true` — use fine-tuned checkpoint instead of reference

### Frontend — Voice Profile UX

- [x] **(P1)** "Set up your voice profile" CTA → resource check → disclosure → record 30 clips → train
- [x] **(P1)** Recording flow:
  - [x] **(P1)** Show prompt text, record button, waveform preview
  - [x] **(P1)** Per-clip validation feedback
  - [x] **(P2)** Progress: clip X of 30
- [x] **(P2)** Review screen — list clips, replay, re-record
- [x] **(P1)** Friendly progress copy during training: "Listening to your voice...", "Almost ready..."
- [x] **(P2)** Preview clip auto-plays when training reaches a usable checkpoint

---

## M3 — Polish, Multilingual, Export

Goal: ship-ready product. Multi-language voices, export/import voice profiles.

- [x] **(P2)** Multi-language support in UI (XTTS supports 17+) — language picker
- [x] **(P2)** Speed/emphasis controls (sliders, not raw params)
- [ ] **(P2)** Export flow:
  - [ ] **(P2)** `POST /projects/{id}/export` — package profile + metadata + license note as `.zip`
  - [ ] **(P2)** UI: "Save voice profile" button → file dialog
  - [ ] **(P2)** Import: drop a `.zip` to restore a project
- [ ] **(P2)** Onboarding: first-run explainer ("Here's what to expect")
- [ ] **(P3)** Error reporter — "Send report" button bundles last log + system profile (opt-in)

---

## M4 — Desktop Packaging (Tauri)

Goal: one-click Windows .exe install. User opens app, records, hears their voice, no terminal needed.

### Sidecar architecture

- [x] **(P1)** FastAPI binds to `127.0.0.1:0` (random free port) and writes the port to a known file → Tauri reads it on startup
- [x] **(P1)** Tauri spawns the sidecar at app start, kills it on app close
- [x] **(P2)** Health-check loop on the frontend so UI shows "starting up..." until backend is ready

### Bundling

- [ ] **(P1)** Decide bundling: Tauri shell + sidecar Python backend (PyInstaller or shiv)
- [ ] **(P1)** Package Python runtime + deps as sidecar binary
- [ ] **(P2)** **Strip dev-only deps** before bundle (current `requirements-lock.txt` has ROS 2 noise — needs clean regen)
- [ ] **(P2)** Bundle FFmpeg statically (or drop it — librosa pipeline doesn't need it)

### First-run experience

- [ ] **(P1)** Model download with progress + resumability:
  - [ ] **(P1)** XTTS v2 (~2 GB)
  - [ ] **(P2)** Whisper base (~150 MB) — only if Voice Profile is used
  - [ ] **(P1)** Clear copy: "Downloading voice engine (one-time, ~2 GB)..."
- [x] **(P1)** Models stored in user data dir, not next to the .exe
- [ ] **(P2)** Resume on interruption (network drop, app crash mid-download)

### Distribution

- [ ] **(P1)** Replace Gradio UI with Tauri + web frontend (React/Svelte/etc.)
- [ ] **(P3)** Code-signing for Windows / macOS (post-MVP)
- [ ] **(P3)** Auto-update channel (post-MVP)

---

## Cross-cutting

### Hardware + Performance

- [x] **(P2)** Document tested configs (GTX 1650 4GB, M3 Pro CPU, CPU-only) in README with expected times for each path
- [x] **(P1)** Memory guard: refuse training if free VRAM < threshold, suggest Quick Clone instead

### Observability

- [ ] **(P3)** Add request IDs to API logs
- [ ] **(P3)** Optional anonymous telemetry (opt-in, off by default)

### Quality

- [ ] **(P1)** Test suite: unit (audio validator, preprocessor) + integration (API smoke tests)
- [ ] **(P2)** CI: lint + test on push (later)
- [ ] **(P2)** README: install, run, troubleshoot (CUDA, FFmpeg, antivirus false positives)

### UX Copy Audit

- [x] **(P2)** Replace any leaked jargon: "epoch" → "round", "loss" → hidden, "checkpoint" → "saved progress"
- [ ] **(P2)** Error messages always action-oriented ("Move closer to the mic" not "SNR below threshold")
- [x] **(P2)** Resource warnings in plain language ("This will use your GPU for ~3 hours") not specs

---

## Open Questions

- [ ] Voice Profile training on Apple Silicon (M-series CPU) — feasible with MPS backend, or refuse?
- [ ] Licensing terms for exported voice models (Coqui XTTS license implications)
- [ ] Should Quick Clone reference clip be 1 prompt or 3 short ones for better quality?
- [ ] First-run model download: bundle XTTS in installer (~2 GB installer) vs download on first run (smaller installer, requires net)?
