"""
sidecar.py — entry point for the packaged VoiceForge sidecar.

Tauri spawns this module. It starts:
  1. The FastAPI backend on a free port (published in the port file)
  2. The Gradio frontend on a separate free port (also published)

Neither port is fixed. On startup we ask the kernel for two free ports
(bind to :0), publish both in the runtime port file, and hand the frontend
port to Gradio via GRADIO_SERVER_PORT. This means the app never depends on
any specific port being available — it adapts to whatever is free.

Both processes share the same Python runtime and run concurrently via asyncio.
"""

from __future__ import annotations

import asyncio
import os
import sys

import uvicorn

from backend.runtime import (
    clear_backend_port_file,
    find_free_port,
    write_backend_port_file,
)


HOST = os.environ.get("VOICEFORGE_BACKEND_HOST", "127.0.0.1")

# Pick free ports ONCE at import time so both coroutines agree on them and we
# can publish them before either server finishes booting. A preferred port can
# be forced via env (used by tests / dev), otherwise the kernel assigns one.
_BACKEND_PREFERRED = os.environ.get("VOICEFORGE_BACKEND_PORT")
_FRONTEND_PREFERRED = os.environ.get("VOICEFORGE_FRONTEND_PORT")

BACKEND_PORT = find_free_port(
    HOST,
    preferred=int(_BACKEND_PREFERRED) if _BACKEND_PREFERRED not in (None, "0", "") else None,
)
FRONTEND_PORT = find_free_port(
    HOST,
    preferred=int(_FRONTEND_PREFERRED) if _FRONTEND_PREFERRED not in (None, "0", "") else None,
)


async def serve_backend() -> None:
    """Run the FastAPI backend and publish both backend + frontend ports."""
    clear_backend_port_file()
    config = uvicorn.Config(
        "backend.main:app",
        host=HOST,
        port=BACKEND_PORT,
        log_level=os.environ.get("VOICEFORGE_BACKEND_LOG_LEVEL", "info"),
    )
    server = uvicorn.Server(config)
    sockets = [config.bind_socket()]
    bound_port = sockets[0].getsockname()[1]

    # Publish BOTH ports up front so the Tauri shell can discover the Gradio
    # URL directly from the port file (no fixed-port assumption anywhere).
    port_file = write_backend_port_file(HOST, bound_port, frontend_port=FRONTEND_PORT)
    print(f"VoiceForge backend listening on http://{HOST}:{bound_port}", flush=True)
    print(f"VoiceForge frontend will run on http://{HOST}:{FRONTEND_PORT}", flush=True)
    print(f"VoiceForge backend port file: {port_file}", flush=True)
    try:
        await server.serve(sockets=sockets)
    finally:
        clear_backend_port_file()


async def serve_frontend() -> None:
    """Launch the Gradio frontend in a subprocess on the chosen free port."""
    # Determine the Python executable — in a PyInstaller bundle it's sys.executable,
    # in dev mode it's the venv python.
    python = sys.executable

    proc = await asyncio.create_subprocess_exec(
        python, "-m", "frontend.app",
        env={
            **os.environ,
            "GRADIO_SERVER_PORT": str(FRONTEND_PORT),
            "VOICEFORGE_FRONTEND_PORT": str(FRONTEND_PORT),
        },
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    print(f"VoiceForge frontend starting on port {FRONTEND_PORT}", flush=True)

    # Stream the child's output so Gradio errors surface in the Tauri console
    # instead of vanishing (previously a silent crash looked like a hang).
    async def _pipe(stream, prefix):
        if stream is None:
            return
        async for line in stream:
            print(f"[{prefix}] {line.decode(errors='replace').rstrip()}", flush=True)

    try:
        await asyncio.gather(
            _pipe(proc.stdout, "gradio"),
            _pipe(proc.stderr, "gradio"),
            proc.wait(),
        )
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
