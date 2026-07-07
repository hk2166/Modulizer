from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from backend.api.inference import router as inference_router
from backend.api.projects import router as projects_router
from backend.api.jobs import router as jobs_router
from backend.api.landing import render_landing_page
from backend.system.hardware import get_full_system_profile
from backend.core.logger import logger
from backend.core.settings import init_runtime_config


app = FastAPI(title="VoiceForge API")

app.include_router(inference_router)
app.include_router(projects_router)
app.include_router(jobs_router)


@app.on_event('startup')
async def startup():
    config = init_runtime_config()
    logger.info(f"Runtime config: low_vram_mode={config.low_vram_mode}, cuda={config.cuda_available}")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def root():
    """
    Landing page — pretty HTML overview of every endpoint.

    `include_in_schema=False` keeps this off /docs and /openapi.json so
    the API spec stays focused on actual API endpoints.
    """
    return render_landing_page()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/system")
def system_info():
    return get_full_system_profile()


@app.get("/models/status")
def models_status():
    """
    Tauri startup shell polls this to decide whether to show a progress screen.

    This is a PURE DISK CHECK — it never triggers a download.
    Downloads happen lazily on first use (Quick Clone / fine-tune).

    Returns:
        ready: True when all required model files exist on disk.
        downloads: always [] — we don't track download progress here because
                   downloads are triggered by inference, not this endpoint.
        xtts_ready: whether XTTS v2 model.pth exists.
        whisper_ready: whether faster-whisper base model exists.
    """
    from pathlib import Path
    from backend.core.settings import MODELS_DIR

    # XTTS v2 — Coqui stores models in ~/.local/share/tts/<slug>/
    xtts_dir = (
        Path.home()
        / ".local/share/tts"
        / "tts_models--multilingual--multi-dataset--xtts_v2"
    )
    # model.pth is the main weight file; its presence means the download finished.
    xtts_ready = (xtts_dir / "model.pth").exists()

    # faster-whisper base — downloaded by transcriber on first transcription
    whisper_base_dir = MODELS_DIR / "whisper" / "models--Systran--faster-whisper-base"
    whisper_ready = whisper_base_dir.exists()

    # We only gate on XTTS — Whisper downloads in the background on first use
    # and its absence shouldn't block the user from reaching the app.
    ready = xtts_ready

    return {
        "ready": ready,
        "downloads": [],          # progress tracking not implemented; shell skips bar
        "xtts_ready": xtts_ready,
        "whisper_ready": whisper_ready,
    }


@app.get("/frontend-url")
def frontend_url():
    """
    Tell the Tauri shell where the Gradio UI is running.
    In production the sidecar starts Gradio on port 7860.
    """
    import os
    port = int(os.environ.get("VOICEFORGE_FRONTEND_PORT", "7860"))
    return {"url": f"http://127.0.0.1:{port}"}
