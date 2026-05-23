"""
client.py — thin HTTP wrapper around the FastAPI backend.

Why a separate file?
────────────────────
The Gradio UI shouldn't know HTTP details (URLs, headers, multipart).
We isolate that here so:
  - UI code stays focused on widgets and state
  - When we swap Gradio → Tauri/React later, only this file needs the
    JS-port equivalent — UI logic carries over conceptually
  - Tests can hit the API directly without spinning up Gradio

All functions raise `BackendError` on failure with a user-friendly message.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import requests


# Where the FastAPI sidecar listens. In Tauri we'll discover this dynamically
# (random port written to a file by the Python sidecar). For dev, fixed.
BASE_URL = "http://localhost:8000"

# Conservative timeouts — generation can be slow on CPU
DEFAULT_TIMEOUT = 30          # short calls (CRUD, status polls)
SYNTH_TIMEOUT = 5 * 60        # XTTS inference can take a minute on CPU


class BackendError(Exception):
    """Raised when the backend returns an error or is unreachable."""


def _get(path: str, **kwargs) -> dict:
    """GET helper. Returns parsed JSON or raises BackendError."""
    try:
        r = requests.get(f"{BASE_URL}{path}", timeout=DEFAULT_TIMEOUT, **kwargs)
    except requests.RequestException as e:
        raise BackendError(f"Couldn't reach the app backend. Is it running? ({e})")
    return _check(r)


def _post(path: str, *, timeout: float = DEFAULT_TIMEOUT, **kwargs) -> dict:
    """POST helper. `kwargs` is forwarded to requests (json=, files=, ...)."""
    try:
        r = requests.post(f"{BASE_URL}{path}", timeout=timeout, **kwargs)
    except requests.RequestException as e:
        raise BackendError(f"Couldn't reach the app backend. Is it running? ({e})")
    return _check(r)


def _check(r: requests.Response) -> dict:
    """Validate response and return JSON payload."""
    if r.status_code >= 400:
        # FastAPI returns errors as {"detail": "..."}; surface that to the user
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text or f"HTTP {r.status_code}"
        raise BackendError(str(detail))
    if r.status_code == 204:
        return {}
    return r.json()


# ── System ────────────────────────────────────────────────────────

def get_system_profile() -> dict:
    """Return hardware info: GPU, RAM, disk, OS."""
    return _get("/system")


def health_check() -> bool:
    """Quick liveness check. Returns True if the backend is up."""
    try:
        _get("/health")
        return True
    except BackendError:
        return False


# ── Projects ──────────────────────────────────────────────────────

def create_project(name: str) -> dict:
    """Create a new voice project. Returns its metadata (incl. id)."""
    return _post("/projects", json={"name": name})


def get_project(project_id: str) -> dict:
    """Fetch project metadata + clip counts."""
    return _get(f"/projects/{project_id}")


# ── Clips ─────────────────────────────────────────────────────────

def upload_clip(project_id: str, wav_path: str | Path) -> dict:
    """
    Upload a recorded .wav for a project.
    Returns: {clip_id, duration_s, sample_rate, valid, errors, warning}
    """
    wav_path = Path(wav_path)
    with open(wav_path, "rb") as f:
        files = {"file": (wav_path.name, f, "audio/wav")}
        return _post(f"/projects/{project_id}/clips", files=files)


def import_recording(project_id: str, source_path: str | Path) -> dict:
    """
    Upload a long audio/video file. Returns: {job_id, status, filename}.
    Poll /jobs/{job_id} until completed; the `result` field contains the
    import summary (segments_kept, clip_ids, ...).
    """
    source_path = Path(source_path)
    # Long imports may take a while to upload (big file) so use a generous timeout
    with open(source_path, "rb") as f:
        # Let requests guess content-type from extension
        files = {"file": (source_path.name, f)}
        return _post(
            f"/projects/{project_id}/import",
            files=files,
            timeout=10 * 60,  # 10 min, mostly for the upload itself
        )


def list_clips(project_id: str) -> list[dict]:
    """List all uploaded clips for a project."""
    return _get(f"/projects/{project_id}/clips")


def delete_clip(project_id: str, clip_id: str) -> None:
    """Remove a clip (used for re-record)."""
    _check(requests.delete(f"{BASE_URL}/projects/{project_id}/clips/{clip_id}",
                           timeout=DEFAULT_TIMEOUT))


# ── Preprocess job ────────────────────────────────────────────────

def start_preprocess(project_id: str) -> dict:
    """Kick off background preprocessing. Returns {job_id, status, clip_count}."""
    return _post(f"/projects/{project_id}/preprocess")


def get_job(job_id: str) -> dict:
    """Poll a job's status / progress / message."""
    return _get(f"/jobs/{job_id}")


# ── Synthesis ─────────────────────────────────────────────────────

def synthesize(project_id: str, text: str, language: str = "en") -> dict:
    """
    Generate speech using the project's reference clip.
    Returns: {output (file path), reference_clip, language}
    """
    return _post(
        f"/projects/{project_id}/synthesize",
        json={"text": text, "language": language},
        timeout=SYNTH_TIMEOUT,
    )


def preview_url(project_id: str, clip_id: str) -> str:
    """URL the UI can hand to an <audio> element to stream a generated clip."""
    return f"{BASE_URL}/projects/{project_id}/preview/{clip_id}"
