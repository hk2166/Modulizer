from pathlib import Path

from dataclasses import dataclass, field

@dataclass
class RuntimeConfig:
    """Runtime flags derived from hardware detection at startup."""
    low_vram_mode: bool = False
    cuda_available: bool = False
    gpu_name: str | None = None
    vram_gb: float = 0.0

# Module-level singleton — populated by init_runtime_config()
runtime_config = RuntimeConfig()


def init_runtime_config() -> RuntimeConfig:
    """Call once at app startup to detect hardware and set runtime flags."""
    from backend.system.hardware import get_gpu_info

    gpu = get_gpu_info()
    runtime_config.cuda_available = gpu.get("cuda", False)
    runtime_config.gpu_name = gpu.get("gpu_name")
    runtime_config.vram_gb = gpu.get("vram_gb", 0.0)
    runtime_config.low_vram_mode = gpu.get("low_vram_mode", False)

    return runtime_config


# Project root is two levels up from this file (backend/core/settings.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = DATA_DIR / "logs"
CACHE_DIR = DATA_DIR / "cache"
EXPORTS_DIR = DATA_DIR / "exports"
TEMP_DIR = DATA_DIR / "temp"
MODELS_DIR = PROJECT_ROOT / "models"

TTS_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
TTS_OUTPUT_DIR = DATA_DIR / "exports"
