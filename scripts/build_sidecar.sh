#!/usr/bin/env bash
# Build the VoiceForge Python sidecar with PyInstaller.
#
# Output: dist/voiceforge-sidecar/voiceforge-sidecar
# Tauri expects it at: src-tauri/binaries/voiceforge-sidecar-<TARGET_TRIPLE>
#
# Usage:
#   ./scripts/build_sidecar.sh              # build only
#   ./scripts/build_sidecar.sh --install    # build + copy into Tauri binaries dir
#
# Requires: venv/bin/pyinstaller (pip install pyinstaller)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

ok()   { printf '\033[32m✓\033[0m %s\n' "$1"; }
info() { printf '\033[36m›\033[0m %s\n' "$1"; }
fail() { printf '\033[31m✗\033[0m %s\n' "$1" >&2; exit 1; }

[[ -d "$REPO_ROOT/venv" ]] || fail "venv missing — run ./scripts/setup.sh first."
source "$REPO_ROOT/venv/bin/activate"

command -v pyinstaller >/dev/null 2>&1 || fail "pyinstaller not found. Run: pip install pyinstaller==6.11.0"

# ── Build ──────────────────────────────────────────────────────────
info "Building Python sidecar…"
pyinstaller voiceforge.spec \
  --noconfirm \
  --clean \
  --distpath dist

BINARY="$REPO_ROOT/dist/voiceforge-sidecar/voiceforge-sidecar"
[[ -f "$BINARY" ]] || fail "Build failed — binary not found at $BINARY"
ok "Sidecar built: $(du -sh "$BINARY" | cut -f1)"

# ── Copy to Tauri binaries dir ─────────────────────────────────────
if [[ "${1:-}" == "--install" ]]; then
    # Detect the Rust target triple
    TARGET=$(rustup show active-toolchain | grep -oP '[\w]+-[\w]+-[\w]+(?=-)' | head -1 || rustc -vV | grep host | awk '{print $2}')
    DEST="$REPO_ROOT/src-tauri/binaries/voiceforge-sidecar-$TARGET"

    info "Copying to $DEST"
    mkdir -p "$(dirname "$DEST")"
    cp -r "$REPO_ROOT/dist/voiceforge-sidecar" "$(dirname "$DEST")/voiceforge-sidecar-$TARGET"
    # Tauri wants a symlink at the exact triple-suffixed path pointing to the exe
    ln -sf "voiceforge-sidecar-$TARGET/voiceforge-sidecar" "$DEST"

    ok "Installed to src-tauri/binaries/"
fi

echo
ok "Done. Run './scripts/build_app.sh' to build the full Tauri installer."
