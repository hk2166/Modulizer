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
 * Check that required models are present on disk.
 *
 * The backend /models/status endpoint is a pure disk check — it never
 * triggers downloads. If models aren't present the user needs to run
 * a Quick Clone first (which lazily downloads XTTS on first synthesis).
 *
 * We poll until ready=true, showing the appropriate status message.
 */
async function ensureModels(backendUrl) {
  let notReadyCount = 0;

  for (;;) {
    let status;
    try {
      const r = await fetch(`${backendUrl}/models/status`, { cache: "no-store" });
      if (!r.ok) return;          // endpoint missing → treat as ready (dev mode)
      status = await r.json();
    } catch {
      // endpoint unreachable → treat as ready so we don't block startup
      return;
    }

    if (status.ready) {
      hideProgress();
      return;
    }

    notReadyCount++;

    // Show download progress if the backend reports active downloads
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
    } else if (!status.xtts_ready) {
      // Models not present — they download on first Quick Clone use.
      // Don't block; let the user reach the app and trigger the download naturally.
      if (notReadyCount >= 2) {
        hideProgress();
        setStatus("Voice engine will download on first use…");
        await sleep(1800);
        return;
      }
      setStatus("Checking voice engine…");
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

    // 3. Get the Gradio URL. Primary source is the port file (via Rust),
    //    which the sidecar populates with a guaranteed-free frontend port.
    //    Falls back to the backend's /frontend-url endpoint.
    let gradioUrl = null;
    for (let i = 0; i < 40 && !gradioUrl; i++) {
      // 3a. Try the port file first (authoritative — no fixed port assumed)
      try {
        gradioUrl = await invoke("frontend_url");
      } catch {
        // frontend port not published yet
      }
      // 3b. Fall back to the backend endpoint
      if (!gradioUrl) {
        try {
          const r = await fetch(`${backendUrl}/frontend-url`, { cache: "no-store" });
          if (r.ok) {
            const data = await r.json();
            if (data.url) gradioUrl = data.url;
          }
        } catch {
          // backend not answering that route yet
        }
      }
      if (!gradioUrl) await sleep(500);
    }

    if (!gradioUrl) {
      throw new Error("Could not determine the app URL. Try restarting.");
    }

    // 4. Wait until Gradio actually answers on that port
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
