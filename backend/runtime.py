from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from backend.core.settings import DATA_DIR


PORT_FILE_ENV = "VOICEFORGE_PORT_FILE"
DEFAULT_PORT_FILE = DATA_DIR / "runtime" / "backend-port.json"


def get_backend_port_file() -> Path:
    """Return the known file used by the sidecar to announce its port."""
    override = os.environ.get(PORT_FILE_ENV)
    return Path(override).expanduser() if override else DEFAULT_PORT_FILE


def write_backend_port_file(host: str, port: int) -> Path:
    """Atomically write the bound backend address for the Tauri shell."""
    port_file = get_backend_port_file()
    port_file.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "host": host,
        "port": port,
        "base_url": f"http://{host}:{port}",
    }
    tmp_file = port_file.with_suffix(f"{port_file.suffix}.tmp")
    tmp_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_file.replace(port_file)
    return port_file


def clear_backend_port_file() -> None:
    """Remove a stale backend port announcement if one exists."""
    port_file = get_backend_port_file()
    try:
        port_file.unlink()
    except FileNotFoundError:
        pass
