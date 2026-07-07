from __future__ import annotations

import json
import os
import socket
from pathlib import Path
from typing import Any

from backend.core.settings import DATA_DIR


PORT_FILE_ENV = "VOICEFORGE_PORT_FILE"
DEFAULT_PORT_FILE = DATA_DIR / "runtime" / "backend-port.json"


def find_free_port(host: str = "127.0.0.1", preferred: int | None = None) -> int:
    """Return a usable TCP port on *host*.

    If *preferred* is given and currently free, it's returned; otherwise the
    OS assigns a random free port (bind to port 0). This is how we avoid ever
    depending on a specific port being available — we ask the kernel for one
    it knows is free, right before we use it.
    """
    if preferred is not None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((host, preferred))
                return preferred
            except OSError:
                pass  # preferred port taken — fall through to random

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, 0))
        return s.getsockname()[1]


def get_backend_port_file() -> Path:
    """Return the known file used by the sidecar to announce its port."""
    override = os.environ.get(PORT_FILE_ENV)
    return Path(override).expanduser() if override else DEFAULT_PORT_FILE


def write_backend_port_file(
    host: str,
    port: int,
    frontend_port: int | None = None,
) -> Path:
    """Atomically write the bound backend + frontend addresses for the shell."""
    port_file = get_backend_port_file()
    port_file.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "host": host,
        "port": port,
        "base_url": f"http://{host}:{port}",
    }
    if frontend_port is not None:
        payload["frontend_port"] = frontend_port
        payload["frontend_url"] = f"http://{host}:{frontend_port}"

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
