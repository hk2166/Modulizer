from fastapi import FastAPI

from backend.api.inference import router as inference_router
from backend.system.hardware import get_full_system_profile
from backend.core.logger import logger
from backend.core.settings import init_runtime_config
from backend.api.projects import router as projects_router


app = FastAPI(title="VoiceForge API")

app.include_router(inference_router)
app.include_router(projects_router)


@app.on_event('startup')
async def startup():
    config = init_runtime_config()
    logger.info(f"Runtime config: low_vram_mode={config.low_vram_mode}, cuda={config.cuda_available}")


@app.get("/")
def root():
    return {"message": "VoiceForge backend running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/system")
def system_info():
    return get_full_system_profile()
