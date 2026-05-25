#!/usr/bin/env bash
# VoiceForge — Linux setup script.
#
# Creates a Python virtualenv, installs torch/torchaudio matched to your
# hardware (CPU or CUDA 12.1), then installs the rest of the requirements.
#
# Usage:
#   ./scripts/setup.sh                # default: auto-detect CPU vs GPU
#   ./scripts/setup.sh --cpu          # force CPU-only torch
#   ./scripts/setup.sh --cuda 12.1    # force a specific CUDA index
#   ./scripts/setup.sh --dev          # also install requirements-dev.txt
#
# Re-runs are safe — pip skips already-satisfied requirements.

set -euo pipefail

# Resolve repo root from this script's location so it works from any cwd.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

VENV_DIR="${REPO_ROOT}/venv"
PY_MIN_MAJOR=3
PY_MIN_MINOR=11
TORCH_VERSION="2.5.1"
DEFAULT_CUDA_TAG="cu121"

# ── Pretty output helpers ─────────────────────────────────────────────────
ok()    { printf '\033[32m✓\033[0m %s\n' "$1"; }
info()  { printf '\033[36m›\033[0m %s\n' "$1"; }
warn()  { printf '\033[33m!\033[0m %s\n' "$1"; }
fail()  { printf '\033[31m✗\033[0m %s\n' "$1" >&2; exit 1; }

# ── Argument parsing ──────────────────────────────────────────────────────
FORCE_CPU=0
FORCE_CUDA_TAG=""
INSTALL_DEV=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cpu)   FORCE_CPU=1; shift ;;
        --cuda)  FORCE_CUDA_TAG="$2"; shift 2 ;;
        --dev)   INSTALL_DEV=1; shift ;;
        -h|--help)
            # Print only the leading docstring (the contiguous block of
            # comment lines at the very top, after the shebang).
            awk '
                NR==1 {next}                                   # skip shebang
                /^#/ { sub(/^# ?/, ""); print; next }
                { exit }
            ' "$0"
            exit 0
            ;;
        *) fail "Unknown option: $1 (try --help)" ;;
    esac
done

# ── Step 1: Python version check ──────────────────────────────────────────
info "Looking for Python ${PY_MIN_MAJOR}.${PY_MIN_MINOR}+"
PY_BIN=""
for cand in python3.13 python3.12 python3.11 python3; do
    if command -v "$cand" >/dev/null 2>&1; then
        ver=$("$cand" -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')")
        major="${ver%%.*}"
        minor="${ver##*.}"
        if [[ "$major" -ge "$PY_MIN_MAJOR" && "$minor" -ge "$PY_MIN_MINOR" ]]; then
            PY_BIN="$cand"
            ok "Found ${cand} (${ver})"
            break
        fi
    fi
done

if [[ -z "$PY_BIN" ]]; then
    fail "Need Python ${PY_MIN_MAJOR}.${PY_MIN_MINOR}+ — install it (e.g. 'sudo apt install python3.11 python3.11-venv') and re-run."
fi

# ── Step 2: System packages we can't pip-install ──────────────────────────
# ffmpeg comes bundled via imageio-ffmpeg, but soundfile/librosa need
# libsndfile and OpenMP. We probe by trying to import soundfile rather
# than reading ldconfig, since ldconfig caches lag behind real installs.
if ! python3 -c "import ctypes; ctypes.CDLL('libsndfile.so.1')" 2>/dev/null; then
    warn "libsndfile not found — audio loading will fail."
    warn "  Debian/Ubuntu: sudo apt install -y libsndfile1 libgomp1"
    warn "  Fedora:        sudo dnf install -y libsndfile libgomp"
    warn "  Arch:          sudo pacman -S libsndfile"
fi

# ── Step 3: venv ──────────────────────────────────────────────────────────
if [[ ! -d "$VENV_DIR" ]]; then
    info "Creating virtualenv at ${VENV_DIR}"
    "$PY_BIN" -m venv "$VENV_DIR"
    ok "venv created"
else
    ok "venv already exists"
fi

# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

info "Upgrading pip"
pip install --quiet --upgrade pip wheel setuptools

# ── Step 4: Decide CPU vs CUDA torch wheel ───────────────────────────────
TORCH_TAG=""
if [[ $FORCE_CPU -eq 1 ]]; then
    TORCH_TAG="cpu"
    info "Forced CPU build of torch"
elif [[ -n "$FORCE_CUDA_TAG" ]]; then
    TORCH_TAG="${FORCE_CUDA_TAG/./}"  # 12.1 -> 121
    [[ "$TORCH_TAG" != cu* ]] && TORCH_TAG="cu${TORCH_TAG}"
    info "Forced CUDA build (${TORCH_TAG})"
elif command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    TORCH_TAG="$DEFAULT_CUDA_TAG"
    info "GPU detected — installing CUDA build (${TORCH_TAG})"
else
    TORCH_TAG="cpu"
    info "No GPU detected — installing CPU build"
fi

TORCH_INDEX="https://download.pytorch.org/whl/${TORCH_TAG}"

# ── Step 5: Install torch + torchaudio from the chosen index ─────────────
info "Installing torch==${TORCH_VERSION}+${TORCH_TAG} torchaudio==${TORCH_VERSION}+${TORCH_TAG}"
pip install --quiet \
    "torch==${TORCH_VERSION}+${TORCH_TAG}" \
    "torchaudio==${TORCH_VERSION}+${TORCH_TAG}" \
    --index-url "$TORCH_INDEX"
ok "torch + torchaudio installed"

# ── Step 6: Install everything else ──────────────────────────────────────
info "Installing runtime requirements"
# We use --no-deps for nothing here on purpose — let pip resolve normally.
# torch/torchaudio are already pinned and won't get re-installed.
pip install --quiet -r requirements.txt
ok "Runtime deps installed"

if [[ $INSTALL_DEV -eq 1 ]]; then
    info "Installing dev/test requirements"
    pip install --quiet -r requirements-dev.txt
    ok "Dev deps installed"
fi

# ── Step 7: Smoke check ──────────────────────────────────────────────────
info "Verifying core imports"
python - <<'PY'
import sys
mods = ["fastapi", "uvicorn", "torch", "torchaudio", "TTS", "trainer",
        "librosa", "soundfile", "numpy", "scipy", "faster_whisper",
        "gradio", "requests", "psutil"]
missing = []
for m in mods:
    try:
        __import__(m)
    except Exception as e:
        missing.append(f"{m}: {e}")
if missing:
    print("\nFailed imports:")
    for line in missing:
        print(f"  - {line}")
    sys.exit(1)
print(f"All {len(mods)} core modules import cleanly.")
PY
ok "All set."

echo
ok "Setup complete."
echo
echo "Next:"
echo "  ./scripts/run.sh        # boot backend + frontend"
echo
