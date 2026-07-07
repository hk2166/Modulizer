#!/usr/bin/env bash
# Build the full VoiceForge desktop app (.deb + .AppImage on Linux).
#
# Steps:
#   1. Build the Python sidecar with PyInstaller
#   2. Install the sidecar into src-tauri/binaries/
#   3. Build the Vite frontend (loading shell)
#   4. Run cargo tauri build
#
# Output: src-tauri/target/release/bundle/
#   - deb/voiceforge_0.1.0_amd64.deb
#   - appimage/voiceforge_0.1.0_amd64.AppImage
#
# Usage:
#   ./scripts/build_app.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

ok()   { printf '\033[32m✓\033[0m %s\n' "$1"; }
info() { printf '\033[36m›\033[0m %s\n' "$1"; }
fail() { printf '\033[31m✗\033[0m %s\n' "$1" >&2; exit 1; }

# ── Step 1: Python sidecar ─────────────────────────────────────────
info "Building Python sidecar…"
./scripts/build_sidecar.sh --install

# ── Step 2: Vite frontend ──────────────────────────────────────────
info "Building Vite loading shell…"
npm run frontend:build

# ── Step 3: Tauri ─────────────────────────────────────────────────
info "Running cargo tauri build…"
cargo tauri build

echo
ok "Build complete."
echo "  Installers: src-tauri/target/release/bundle/"
ls src-tauri/target/release/bundle/*/* 2>/dev/null | grep -E "\.deb|\.AppImage" | while read f; do
    echo "    $(du -sh "$f" | cut -f1)  $f"
done
