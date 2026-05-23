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
