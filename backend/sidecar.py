from __future__ import annotations

import asyncio
import os

import uvicorn

from backend.runtime import clear_backend_port_file, write_backend_port_file


HOST = os.environ.get("VOICEFORGE_BACKEND_HOST", "127.0.0.1")
PORT = int(os.environ.get("VOICEFORGE_BACKEND_PORT", "0"))


async def serve() -> None:
    """
    Bind FastAPI to localhost on a random free port and publish that port.

    Tauri starts this module as the Python sidecar. Binding before writing the
    file prevents the frontend from discovering a port that is not listening.
    """
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


def main() -> None:
    asyncio.run(serve())


if __name__ == "__main__":
    main()
