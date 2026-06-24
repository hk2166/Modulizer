import { invoke } from "@tauri-apps/api/core";

const statusMark = document.querySelector("#statusMark");
const statusText = document.querySelector("#statusText");
const workspace = document.querySelector("#workspace");
const backendDetails = document.querySelector("#backendDetails");
const healthButton = document.querySelector("#healthButton");

let backendUrl = null;

function setStatus(state, text) {
  statusMark.dataset.state = state;
  statusText.textContent = text;
}

async function discoverBackend() {
  backendUrl = await invoke("backend_url");
  const response = await fetch(`${backendUrl}/health`, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Backend returned HTTP ${response.status}`);
  }
  return response.json();
}

async function waitForBackend() {
  setStatus("starting", "Starting up...");
  for (;;) {
    try {
      const health = await discoverBackend();
      setStatus("ready", "Ready");
      workspace.hidden = false;
      backendDetails.textContent = JSON.stringify(
        { backendUrl, health },
        null,
        2,
      );
      return;
    } catch {
      await new Promise((resolve) => setTimeout(resolve, 500));
    }
  }
}

healthButton.addEventListener("click", async () => {
  try {
    const health = await discoverBackend();
    backendDetails.textContent = JSON.stringify({ backendUrl, health }, null, 2);
  } catch (error) {
    backendDetails.textContent = String(error);
  }
});

waitForBackend();
