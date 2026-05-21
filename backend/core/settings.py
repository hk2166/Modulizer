import os
import platform
from pathlib import Path
from dataclasses import dataclass


# ── User data directory ───────────────────────────────────────────
# Returns the OS-standard location for user app data.
# This is where projects, models, logs, and cache live.
# Using the OS standard means:
#   - Data survives app upgrades (not next to the .exe)
#   - No permission issues on Windows (Program Files is read-only)
#   - Antivirus doesn't flag writes to AppData

def get_user_data_dir() -> Path:
    """
    Return the OS-appropriate user data directory for VoiceForge.

    Windows : %APPDATA%/VoiceForge          (e.g. C:/Users/you/AppData/Roaming/VoiceForge)
    macOS   : ~/Library/Application Support/VoiceForge
    Linux   : ~/.local/share/voiceforge

    Falls back to a local `data/` folder next to the project root
    during development (when the env var VOICEFORGE_DEV=1 is set,
    or when running directly from the source tree without packaging).
    """
    # Dev override — keeps data local during development
    if os.environ.get("VOICEFORGE_DEV") == "1":
        return Path(__file__).resolve().parent.parent.parent / "data"

    system = platform.system()

    if system == "Windows":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
        return base / "VoiceForge"

    if system == "Darwin":  # macOS
        return Path.home() / "Library" / "Application Support" / "VoiceForge"

    # Linux and everything else
    xdg_data = os.environ.get("XDG_DATA_HOME", "")
    base = Path(xdg_data) if xdg_data else Path.home() / ".local" / "share"
    return base / "voiceforge"


# ── RuntimeConfig ─────────────────────────────────────────────────
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


# ── Paths ─────────────────────────────────────────────────────────
# Project root — only used for source-relative things (scripts, etc.)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# All user-facing data lives under get_user_data_dir()
DATA_DIR = get_user_data_dir()
LOGS_DIR = DATA_DIR / "logs"
CACHE_DIR = DATA_DIR / "cache"
EXPORTS_DIR = DATA_DIR / "exports"
TEMP_DIR = DATA_DIR / "temp"
MODELS_DIR = DATA_DIR / "models"      # models live in user data, not next to the exe

TTS_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
TTS_OUTPUT_DIR = DATA_DIR / "exports"

SCRIPTS_DIR = PROJECT_ROOT / "data" / "scripts"   # read-only app asset, stays in source tree
DEFAULT_PROMPTS_FILE = SCRIPTS_DIR / "default_prompts.json"
