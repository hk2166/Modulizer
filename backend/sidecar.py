"""
sidecar.py — entry point for the packaged VoiceForge sidecar.

Tauri spawns this module. It starts:
  1. The FastAPI backend on a random port (writes port file for Tauri to discover)
  2. The Gradio frontend on port 7860 (fixed port, read by /frontend-url endpoint)

Both processes share the same Python runtime and run concurrently via asyncio.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path

import uvicorn

from backend.runtime import clear_backend_port_file, write_backend_port_file


HOST = os.environ.get("VOICEFORGE_BACKEND_HOST", "127.0.0.1")
PORT = int(os.environ.get("VOICEFORGE_BACKEND_PORT", "0"))
FRONTEND_PORT = int(os.environ.get("VOICEFORGE_FRONTEND_PORT", "7860"))


async def serve_backend() -> None:
    """Run the FastAPI backend on a random port and publish it."""
    clear_backend_port_file()
    config = uvicorn.Config(
        "backend.main:app",
        host=HOST,
        port=PORT,
        log_level=os.environ.get("VOICEFORGE_BACKEND_LOG_LEVEL", "info"),
    )
    server = uvicorn.Server(config)
    sockets = [config.bind_socket()]
    bound_port = sockets[0].getsockname()[1]
    port_file = write_backend_port_file(HOST, bound_port)
    print(f"VoiceForge backend listening on http://{HOST}:{bound_port}", flush=True)
    print(f"VoiceForge backend port file: {port_file}", flush=True)
    try:
        await server.serve(sockets=sockets)
    finally:
        clear_backend_port_file()


async def serve_frontend() -> None:
    """Launch the Gradio frontend in a subprocess and wait for it."""
    # Kill any stale process that's already holding FRONTEND_PORT so that a
    # previous cargo-tauri-dev session doesn't block the new one.
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex(("127.0.0.1", FRONTEND_PORT)) == 0:
            # Something is already on the port.  Find the pid and kill it.
            try:
                import subprocess as _sp
                result = _sp.run(
                    ["fuser", "-k", f"{FRONTEND_PORT}/tcp"],
                    capture_output=True,
                )
                await asyncio.sleep(1.5)   # give the old process a moment to die
            except Exception:
                pass  # fuser not available — Gradio will pick a new port

    # Determine the Python executable — in a PyInstaller bundle it's sys.executable,
    # in dev mode it's the venv python.
    python = sys.executable

    proc = await asyncio.create_subprocess_exec(
        python, "-m", "frontend.app",
        env={**os.environ, "GRADIO_SERVER_PORT": str(FRONTEND_PORT)},
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    print(f"VoiceForge frontend starting on port {FRONTEND_PORT}", flush=True)
    try:
        await proc.communicate()
    finally:
        try:
            proc.terminate()
        except ProcessLookupError:
            pass


async def serve() -> None:
    """Run both backend and frontend concurrently."""
    await asyncio.gather(serve_backend(), serve_frontend())


def main() -> None:
    asyncio.run(serve())


if __name__ == "__main__":
    main()
