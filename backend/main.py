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
    Tauri startup shell polls this to show download progress on first run.

    Returns:
        ready: True if all required models are present on disk.
        downloads: list of in-progress downloads (empty if ready or not started).
    """
    from pathlib import Path
    from backend.core.settings import MODELS_DIR
    from TTS.utils.manage import ModelManager

    downloads = []

    # Check XTTS v2 (~2 GB)
    try:
        manager = ModelManager()
        model_path, _, _ = manager.download_model(
            "tts_models/multilingual/multi-dataset/xtts_v2",
        )
        xtts_ready = Path(model_path, "model.pth").exists()
    except Exception:
        xtts_ready = False

    # Check Whisper base (~150 MB, used for transcription QA)
    whisper_base_dir = MODELS_DIR / "whisper" / "models--Systran--faster-whisper-base"
    whisper_ready = whisper_base_dir.exists()

    ready = xtts_ready and whisper_ready

    return {
        "ready": ready,
        "downloads": downloads,
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
