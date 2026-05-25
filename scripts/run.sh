#!/usr/bin/env bash
# VoiceForge — Linux run script.
#
# Boots the FastAPI backend (uvicorn) and the Gradio frontend side by side.
# Ctrl+C stops both cleanly.
#
# Usage:
#   ./scripts/run.sh              # both backend + frontend
#   ./scripts/run.sh --backend    # backend only
#   ./scripts/run.sh --frontend   # frontend only (assumes backend is running)
#   ./scripts/run.sh --reload     # uvicorn --reload for backend dev
#
# Environment overrides:
#   BACKEND_PORT   default 8000
#   FRONTEND_PORT  default 7860
#   VOICEFORGE_DEV=1 keeps user data in repo's data/ folder instead of
#       ~/.local/share/voiceforge — useful while developing.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

VENV_DIR="${REPO_ROOT}/venv"
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-7860}"

ok()   { printf '\033[32m✓\033[0m %s\n' "$1"; }
info() { printf '\033[36m›\033[0m %s\n' "$1"; }
fail() { printf '\033[31m✗\033[0m %s\n' "$1" >&2; exit 1; }

[[ -d "$VENV_DIR" ]] || fail "venv missing — run ./scripts/setup.sh first."
# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

RUN_BACKEND=1
RUN_FRONTEND=1
RELOAD=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend)  RUN_FRONTEND=0; shift ;;
        --frontend) RUN_BACKEND=0;  shift ;;
        --reload)   RELOAD=1; shift ;;
        -h|--help)
            awk '
                NR==1 {next}
                /^#/ { sub(/^# ?/, ""); print; next }
                { exit }
            ' "$0"
            exit 0
            ;;
        *) fail "Unknown option: $1 (try --help)" ;;
    esac
done

# Track child pids so we can kill cleanly on Ctrl+C.
PIDS=()
shutdown() {
    info "Shutting down"
    for pid in "${PIDS[@]:-}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        fi
    done
    # Give children a moment to exit, then SIGKILL stragglers.
    sleep 0.5
    for pid in "${PIDS[@]:-}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -9 "$pid" 2>/dev/null || true
        fi
    done
    exit 0
}
trap shutdown INT TERM

# ── Backend ──────────────────────────────────────────────────────────────
if [[ $RUN_BACKEND -eq 1 ]]; then
    UVICORN_ARGS=(
        backend.main:app
        --host 127.0.0.1
        --port "$BACKEND_PORT"
        --log-level info
    )
    [[ $RELOAD -eq 1 ]] && UVICORN_ARGS+=(--reload)

    info "Starting backend on http://127.0.0.1:${BACKEND_PORT}"
    python -m uvicorn "${UVICORN_ARGS[@]}" &
    PIDS+=("$!")
    ok "Backend pid=$!"
fi

# ── Wait for backend to come up (so frontend probe succeeds) ─────────────
if [[ $RUN_BACKEND -eq 1 && $RUN_FRONTEND -eq 1 ]]; then
    info "Waiting for backend to be ready..."
    for _ in {1..30}; do
        if curl -fsS "http://127.0.0.1:${BACKEND_PORT}/health" >/dev/null 2>&1; then
            ok "Backend is ready"
            break
        fi
        sleep 0.5
    done
fi

# ── Frontend ─────────────────────────────────────────────────────────────
if [[ $RUN_FRONTEND -eq 1 ]]; then
    info "Starting frontend on http://127.0.0.1:${FRONTEND_PORT}"
    GRADIO_SERVER_PORT="$FRONTEND_PORT" python -m frontend.app &
    PIDS+=("$!")
    ok "Frontend pid=$!"
fi

# Wait on the first child to exit; if any of them dies we tear everything
# down so you don't end up with a stranded backend after a frontend crash.
wait -n "${PIDS[@]:-}"
shutdown
