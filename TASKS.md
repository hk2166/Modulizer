# VoiceForge — Tasks

Local-first desktop app for AI voice cloning. Built around XTTS v2, FastAPI, and Gradio (MVP) → Tauri (later).

**Guiding principle:** UX is the product. No ML jargon ever leaks to the user. "Set up your voice profile" — not "train a model."

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

- [x] XTTS v2 loads and generates audio (`test_xtts.py`)
- [x] FastAPI app scaffolded (`main.py`, inference router, job manager)
- [x] `inference_service.py` working (voice cloning + built-in speaker fallback)
- [x] Hardware detection (`backend/system/hardware.py`)
- [x] **M0 gate:** `uvicorn backend.main:app` boots cleanly
- [x] **M0 gate:** `GET /health` returns 200
- [x] **M0 gate:** `GET /system` returns valid GPU/RAM/disk profile
- [x] **M0 gate:** `POST /tts` generates a valid `.wav` end-to-end
- [x] Fix CUDA/torchaudio mismatch (libcudart.so.13) — picked CPU build
- [x] Wire `low_vram_mode` from `hardware.py` into a runtime config flag
- [x] Update `.gitignore` (pycache, data/cache, data/temp, models, .env)
- [ ] Add `pytest` smoke tests for `/health`, `/system`, `/tts`
- [x] **User data directory helper** — `get_user_data_dir()` returns `%APPDATA%/VoiceForge` on Windows, `~/Library/Application Support/VoiceForge` on macOS, `~/.local/share/voiceforge` on Linux. Replace all uses of `DATA_DIR` with this. Cheap now, painful later.
- [x] Regenerate `requirements-lock.txt` from a clean venv (current lock has ROS 2 packages from dev machine)

---

## M1 — Quick Voice Clone (Primary product path)

Goal: user installs the app, records one clip, types text, hears their cloned voice. Works on any hardware. **This is the default and only path most users will use.**

### Backend — `backend/audio/`

- [x] `recorder.py` — accept uploaded `.wav` chunks, save to `data/projects/{project}/raw/`
- [x] `validator.py` — per-clip checks:
  - [x] Duration in range (3–15s)
  - [x] SNR / silence ratio threshold
  - [x] Clipping detection (peak > -1 dBFS)
  - [x] Sample rate normalization target (24 kHz)
  - [x] User-friendly error messages (no jargon)
- [x] `preprocessor.py` — librosa pipeline:
  - [x] Resample to 24 kHz mono
  - [x] Trim leading/trailing silence
  - [x] Peak normalize loudness
  - [x] Optional spectral denoise pass
  - [x] Output to `data/projects/{project}/processed/`
- [x] `transcriber.py` — faster-whisper for clip QA (verify spoken text matches script)
- [x] Recording prompts asset (`data/scripts/default_prompts.json`) — 30 EN + 30 HI phonetically diverse prompts
- [x] `cleaner.py` — synthesis-targeted reference cleaning (DC offset, high-pass @ 80 Hz, RMS normalize, soft limiter). Off by default — XTTS clones better from natural processed audio.
- [x] `importer.py` — long audio/video import via bundled ffmpeg; concatenates speech bursts and chunks into uniform ~10s clips (handles both continuous and spotty source material).

### API — Project lifecycle

- [x] `POST /projects` — create a new voice project
- [x] `GET /projects/{id}` — project state, clip count, validation status
- [x] `POST /projects/{id}/clips` — upload a clip (multipart)
- [x] `GET /projects/{id}/clips` — list all clips for a project
- [x] `DELETE /projects/{id}/clips/{clip_id}` — re-record support
- [x] `POST /projects/{id}/preprocess` — kick off async preprocess job
- [x] `GET /jobs/{job_id}` — poll job status / progress / result
- [x] All long-running work wired through `JobManager` (background tasks)
- [x] `POST /projects/{id}/import` — multipart upload of long audio/video; segments and saves as project clips (background job)

### API — Reference cloning (synth)

- [x] `POST /projects/{id}/synthesize` — text + reference clip → generated `.wav`
  - Picks the first valid processed clip as the reference automatically (user shouldn't choose one)
  - Accepts `text` and `language`
  - Returns generated audio path
- [x] `GET /projects/{id}/preview/{clip_id}` — stream a generated `.wav` back to the UI

### Frontend — Gradio MVP

- [x] App shell with project picker + create-new flow
- [x] **First-run hardware check** — "Your machine is ready" / "Compatibility notice for fine-tuning" (Quick Clone always available)
- [x] **Quick Clone flow** (the main UX):
  - [x] Show one prompt, record button, waveform preview
  - [x] Inline validation feedback ("Try again — too quiet" / "Looks great")
  - [x] Auto-preprocess on validation pass
  - [x] Text input → generate → playback
- [x] Status panel — friendly progress messages, no percentages of loss
- [x] **Long audio / video import** — drop a podcast, interview, or video; auto-segment into 10s clips usable as references (handles spotty speech via concatenate-then-chunk)

### Persistence

- [ ] SQLite schema via SQLAlchemy: `projects`, `clips`, `jobs`
- [ ] Migration / init on first run
- [ ] Job state persisted to SQLite (so closing the app doesn't lose status)

---

## M2 — Voice Profile (Advanced, fine-tuning)

Goal: power users with capable hardware can fine-tune XTTS on 30 clips for higher fidelity. **Strictly opt-in, hardware-gated, with up-front time and resource warnings.**

### UX — Resource gating

- [x] **Pre-flight resource check** before kickoff:
  - [x] Free VRAM ≥ 3 GB → allow with low-VRAM config
  - [x] Free VRAM ≥ 8 GB → allow standard config
  - [x] No GPU → refuse with: "Voice Profile training needs a graphics card. You can use Quick Clone instead — it works without one."
  - [x] Free disk < 5 GB → refuse with disk-cleanup suggestion
- [ ] **Pre-training disclosure modal** showing:
  - [x] Estimated time based on hardware (e.g. "About 3 hours on your GTX 1650")
  - [x] Estimated power use ("Your GPU will run at full load")
  - [x] What gets saved and where
  - [ ] "I understand, start training" button
- [ ] Background-safe: user can close Tauri window, training continues, status restored on relaunch
- [x] ETA estimation based on hardware profile, updated each round

### Pipeline — `backend/pipelines/training.py`

- [x] Dataset builder — convert processed clips + transcripts into XTTS training format
- [x] Auto-config based on `hardware.py`:
  - [x] GPU + ≥8 GB VRAM → standard config
  - [x] GPU + 3–8 GB VRAM → low-VRAM (fp16, gradient checkpointing, batch_size=2)
  - [x] CPU only → refuse with friendly message (no cloud fallback in v1)
- [x] Training loop wrapper with progress callbacks → `JobManager.update_progress`
  - **Quality follow-ups discovered during this work:**
  - [x] Switch `preprocessor.TARGET_SAMPLE_RATE` from 24000 → 22050 to align with XTTS GPT/dvae native rate. Output stays 24 kHz via the HiFi-GAN decoder. Removes a per-step torchaudio resample.
  - [x] Make `dataset_builder.py` pick a "golden reference" clip (5–7s, near-median pitch + RMS) and store its id in `manifest.json`; teach `project_service.get_reference_clip` to prefer that over `sorted(...)[0]`.
  - [x] Audit `preprocessor._spectral_denoise` — the path is off by default but spectral subtraction strips formants when used; consider replacing with a no-op or a learned model, not a hand-rolled gate.
  - **Hindi (and other Indic-language) pronunciation gap:** XTTS's `tokenizer.preprocess_text` falls through to `basic_cleaners` (lowercase + whitespace) for Hindi — the maintainer left a `# @manmay will implement this` TODO. Same fallback hits Bengali, Tamil, Telugu, Marathi, Gujarati, Punjabi, Urdu. Two ways to close the gap, pick one or do both:
  - [ ] **App-layer pre-normalizer (preferred):** add a `backend/audio/text_cleaners.py` with a `hindi_cleaners(text)` that handles schwa-deletion, halant rules, number expansion, and Latin→Devanagari for inevitable English-leak words. Run it inside `inference_service.generate_speech` and `dataset_builder._sanitize_text` before either path hits the model. Libraries to evaluate: `indic-transliteration`, `aksharamukha`. Survives pip upgrades and stays under our control.
  - [ ] **Venv monkey-patch (fallback):** at app startup, replace `TTS.tts.layers.xtts.tokenizer.VoiceBpeTokenizer.preprocess_text` with a wrapper that routes `lang="hi"` through our `hindi_cleaners` before the base call. Brittle (won't survive `pip install -U TTS`), but catches paths inside Coqui that bypass our service layer (training dataset loader, inference_stream, etc.). Only worth doing if option A leaves measurable gaps.
- [x] Cooperative cancellation — training checks a flag every N batches
- [x] Checkpoint saving to `data/projects/{id}/checkpoints/`
- [ ] Early-stopping / best-checkpoint selection
- [ ] Validation sample synthesis after each round (for UI preview)
- [ ] Crash-safe checkpointing — resume from last good state

### API

- [ ] `POST /projects/{id}/train` — start training job (returns job_id)
- [ ] `POST /jobs/{id}/cancel` — clean cooperative cancellation
- [ ] `POST /projects/{id}/synthesize?profile=true` — use fine-tuned checkpoint instead of reference

### Frontend — Voice Profile UX

- [ ] "Set up your voice profile" CTA → resource check → disclosure → record 30 clips → train
- [ ] Recording flow:
  - [ ] Show prompt text, record button, waveform preview
  - [ ] Per-clip validation feedback
  - [ ] Progress: clip X of 30
- [ ] Review screen — list clips, replay, re-record
- [ ] Friendly progress copy during training: "Listening to your voice...", "Almost ready..."
- [ ] Preview clip auto-plays when training reaches a usable checkpoint

---

## M3 — Polish, Multilingual, Export

Goal: ship-ready product. Multi-language voices, export/import voice profiles.

- [x] Multi-language support in UI (XTTS supports 17+) — language picker
- [ ] Speed/emphasis controls (sliders, not raw params)
- [ ] Export flow:
  - [ ] `POST /projects/{id}/export` — package profile + metadata + license note as `.zip`
  - [ ] UI: "Save voice profile" button → file dialog
  - [ ] Import: drop a `.zip` to restore a project
- [ ] Onboarding: first-run explainer ("Here's what to expect")
- [ ] Error reporter — "Send report" button bundles last log + system profile (opt-in)

---

## M4 — Desktop Packaging (Tauri)

Goal: one-click Windows .exe install. User opens app, records, hears their voice, no terminal needed.

### Sidecar architecture

- [ ] FastAPI binds to `127.0.0.1:0` (random free port) and writes the port to a known file → Tauri reads it on startup
- [ ] Tauri spawns the sidecar at app start, kills it on app close
- [ ] Health-check loop on the frontend so UI shows "starting up..." until backend is ready

### Bundling

- [ ] Decide bundling: Tauri shell + sidecar Python backend (PyInstaller or shiv)
- [ ] Package Python runtime + deps as sidecar binary
- [ ] **Strip dev-only deps** before bundle (current `requirements-lock.txt` has ROS 2 noise — needs clean regen)
- [ ] Bundle FFmpeg statically (or drop it — librosa pipeline doesn't need it)

### First-run experience

- [ ] Model download with progress + resumability:
  - [ ] XTTS v2 (~2 GB)
  - [ ] Whisper base (~150 MB) — only if Voice Profile is used
  - [ ] Clear copy: "Downloading voice engine (one-time, ~2 GB)..."
- [ ] Models stored in user data dir, not next to the .exe
- [ ] Resume on interruption (network drop, app crash mid-download)

### Distribution

- [ ] Replace Gradio UI with Tauri + web frontend (React/Svelte/etc.)
- [ ] Code-signing for Windows / macOS (post-MVP)
- [ ] Auto-update channel (post-MVP)

---

## Cross-cutting

### Hardware + Performance

- [x] Document tested configs (GTX 1650 4GB, M3 Pro CPU, CPU-only) in README with expected times for each path
- [x] Memory guard: refuse training if free VRAM < threshold, suggest Quick Clone instead

### Observability

- [ ] Add request IDs to API logs
- [ ] Optional anonymous telemetry (opt-in, off by default)

### Quality

- [ ] Test suite: unit (audio validator, preprocessor) + integration (API smoke tests)
- [ ] CI: lint + test on push (later)
- [ ] README: install, run, troubleshoot (CUDA, FFmpeg, antivirus false positives)

### UX Copy Audit

- [x] Replace any leaked jargon: "epoch" → "round", "loss" → hidden, "checkpoint" → "saved progress"
- [ ] Error messages always action-oriented ("Move closer to the mic" not "SNR below threshold")
- [x] Resource warnings in plain language ("This will use your GPU for ~3 hours") not specs

---

## Open Questions

- [ ] Voice Profile training on Apple Silicon (M-series CPU) — feasible with MPS backend, or refuse?
- [ ] Licensing terms for exported voice models (Coqui XTTS license implications)
- [ ] Should Quick Clone reference clip be 1 prompt or 3 short ones for better quality?
- [ ] First-run model download: bundle XTTS in installer (~2 GB installer) vs download on first run (smaller installer, requires net)?