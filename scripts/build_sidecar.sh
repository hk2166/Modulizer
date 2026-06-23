#!/usr/bin/env bash
# Build the Python FastAPI sidecar executable expected by Tauri.
#
# Requires pyinstaller in the active environment:
#   pip install pyinstaller

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

TARGET_TRIPLE="$(rustc --print host-tuple 2>/dev/null || rustc -Vv | awk '/host:/ {print $2}')"
[[ -n "$TARGET_TRIPLE" ]] || {
    echo "Could not determine Rust target triple" >&2
    exit 1
}

python -m PyInstaller \
    --clean \
    --onefile \
    --name "voiceforge-sidecar-${TARGET_TRIPLE}" \
    backend/sidecar.py

mkdir -p src-tauri/binaries
cp "dist/voiceforge-sidecar-${TARGET_TRIPLE}" "src-tauri/binaries/voiceforge-sidecar-${TARGET_TRIPLE}"
