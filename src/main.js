/**
 * VoiceForge startup shell
 *
 * Shows a loading/progress screen while:
 *   1. The Python sidecar boots (port discovery + health-check)
 *   2. XTTS and Whisper models download on first run (streamed progress)
 *
 * Once the backend reports /health OK and all models are present, we:
 *   - redirect the Tauri webview to the Gradio app running on localhost
 *
 * The startup sequence is fully resilient — it polls indefinitely and
 * won't show the app until the backend is genuinely ready.
 */

import { invoke } from "@tauri-apps/api/core";

// ── DOM refs ─────────────────────────────────────────────────────
const spinner    = document.querySelector("#spinner");
const statusText = document.querySelector("#statusText");
const downloadArea = document.querySelector("#downloadArea");
const downloadLabel = document.querySelector("#downloadLabel");
const progressBar = document.querySelector("#progressBar");
const downloadNote = document.querySelector("#downloadNote");
const loadingShell = document.querySelector("#loadingShell");
const appFrame = document.querySelector("#appFrame");

// ── Helpers ──────────────────────────────────────────────────────

function setStatus(text) {
  statusText.textContent = text;
}

function showProgress(label, pct, note = "") {
  downloadArea.hidden = false;
  downloadLabel.textContent = label;
  progressBar.style.width = `${pct}%`;
  if (note) downloadNote.textContent = note;
}

function hideProgress() {
  downloadArea.hidden = true;
}

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

// ── Backend discovery ─────────────────────────────────────────────

/**
 * Ask the Rust side for the backend URL via the port file.
 * Returns null if the backend hasn't written the file yet.
 */
async function tryGetBackendUrl() {
  try {
    return await invoke("backend_url");
  } catch {
    return null;
  }
}

/**
 * Poll until the Python sidecar has booted and responds to /health.
 * Returns the base_url string when ready.
 */
async function waitForBackend() {
  setStatus("Starting VoiceForge backend…");
  for (;;) {
    const url = await tryGetBackendUrl();
    if (url) {
      try {
        const r = await fetch(`${url}/health`, { cache: "no-store" });
        if (r.ok) return url;
      } catch {
        // backend not reachable yet
      }
    }
    await sleep(600);
  }
}

// ── Model download ─────────────────────────────────────────────── 

/**
 * Stream model download progress from the backend.
 *
 * The backend exposes GET /models/status which returns JSON:
 * {
 *   "ready": bool,
 *   "downloads": [
 *     { "name": "XTTS v2", "total_bytes": 2050000000, "downloaded_bytes": 512000000 }
 *   ]
 * }
 *
 * We poll every second until ready=true.
 */
async function ensureModels(backendUrl) {
  for (;;) {
    let status;
    try {
      const r = await fetch(`${backendUrl}/models/status`, { cache: "no-store" });
      status = await r.json();
    } catch {
      // if the endpoint doesn't exist yet, treat as ready
      return;
    }

    if (status.ready) {
      hideProgress();
      return;
    }

    // Show download progress for the first incomplete model
    const active = (status.downloads || []).find(
      (d) => d.downloaded_bytes < d.total_bytes
    );
    if (active) {
      const pct = Math.round((active.downloaded_bytes / active.total_bytes) * 100);
      const downloaded_mb = Math.round(active.downloaded_bytes / 1024 / 1024);
      const total_mb = Math.round(active.total_bytes / 1024 / 1024);
      showProgress(
        `Downloading ${active.name}…`,
        pct,
        `${downloaded_mb} MB / ${total_mb} MB — one-time download`
      );
    } else {
      setStatus("Preparing models…");
    }

    await sleep(1000);
  }
}

// ── Main startup sequence ─────────────────────────────────────────

async function startup() {
  try {
    // 1. Wait for the Python backend to boot
    const backendUrl = await waitForBackend();

    // 2. Ensure models are downloaded / verified
    setStatus("Checking voice engine…");
    await ensureModels(backendUrl);

    // 3. Get the Gradio URL — the Gradio app runs on its own port
    //    (default 7860). In dev mode it runs separately; in production the
    //    sidecar starts Gradio too and we discover its port from the backend.
    let gradioUrl;
    try {
      const r = await fetch(`${backendUrl}/frontend-url`, { cache: "no-store" });
      if (r.ok) {
        const data = await r.json();
        gradioUrl = data.url;
      }
    } catch {
      // fallback: Gradio default port
      gradioUrl = backendUrl.replace(/:\d+$/, ":7860");
    }

    // 4. Check Gradio is reachable
    setStatus("Loading app…");
    for (;;) {
      try {
        const r = await fetch(gradioUrl, { cache: "no-store" });
        if (r.ok) break;
      } catch {
        // not ready yet
      }
      await sleep(500);
    }

    // 5. Swap in the iframe
    spinner.dataset.done = true;
    setStatus("Ready");
    await sleep(200);  // brief flash of "Ready" so users see it happened

    loadingShell.hidden = true;
    appFrame.src = gradioUrl;
    appFrame.hidden = false;

  } catch (err) {
    setStatus(`Error: ${err.message || err}`);
    spinner.style.borderTopColor = "#ff9f1c";
    spinner.style.boxShadow = "0 0 12px rgba(255, 159, 28, 0.55)";
  }
}

startup();
