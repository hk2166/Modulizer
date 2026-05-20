import logging
from pathlib import Path

# Absolute path — works regardless of working directory
_LOG_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)

_LOG_FILE = _LOG_DIR / "voiceforge.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(_LOG_FILE),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger("voiceforge")
