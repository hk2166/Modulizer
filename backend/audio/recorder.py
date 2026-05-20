"""
recorder.py — Accept uploaded .wav chunks and save to project raw directory.

HOW IT WORKS:
─────────────
1. User uploads a .wav file via the API (multipart form data).
2. This module validates it's actually a .wav file (basic header check).
3. Saves it to: data/projects/{project_id}/raw/{clip_id}.wav
4. Returns the saved path + clip metadata.

The clip_id is a UUID so filenames never collide, even if the user
re-records the same prompt multiple times.
"""

from pathlib import Path
from uuid import uuid4

import soundfile as sf

from backend.core.logger import logger
from backend.core.settings import DATA_DIR


class RecorderError(Exception):
    """Raised when a recording can't be saved."""
    pass


def get_project_raw_dir(project_id: str) -> Path:
    """
    Get (and create) the raw audio directory for a project.

    Layout:
        data/projects/{project_id}/raw/
    """
    raw_dir = DATA_DIR / "projects" / project_id / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir


def save_clip(project_id: str, audio_bytes: bytes, filename: str | None = None) -> dict:
    """
    Save an uploaded .wav clip to the project's raw directory.

    Args:
        project_id:   The project this clip belongs to.
        audio_bytes:  Raw bytes of the uploaded .wav file.
        filename:     Original filename (for logging only). Optional.

    Returns:
        dict with:
            - clip_id:    Unique identifier for this clip
            - path:       Absolute path where the file was saved
            - duration_s: Duration in seconds
            - sample_rate: Sample rate of the saved file
            - channels:   Number of audio channels

    Raises:
        RecorderError: If the file isn't valid audio or can't be saved.
    """
    clip_id = str(uuid4())
    raw_dir = get_project_raw_dir(project_id)
    output_path = raw_dir / f"{clip_id}.wav"

    # ── Step 1: Write bytes to disk ───────────────────────────────
    # We write first, then validate with soundfile. This is simpler
    # than trying to parse from memory (soundfile can read from path).
    try:
        output_path.write_bytes(audio_bytes)
    except OSError as e:
        raise RecorderError(f"Couldn't save the recording: {e}") from e

    # ── Step 2: Verify it's a valid audio file ────────────────────
    # soundfile.info() reads the header without loading all samples.
    # If the file is corrupt or not audio, this throws.
    try:
        info = sf.info(str(output_path))
    except Exception as e:
        # Clean up the invalid file
        output_path.unlink(missing_ok=True)
        raise RecorderError(
            "That file doesn't look like valid audio. "
            "Please upload a .wav file recorded from your microphone."
        ) from e

    logger.info(
        f"Clip saved: project={project_id}, clip={clip_id}, "
        f"duration={info.duration:.1f}s, sr={info.samplerate}"
    )

    return {
        "clip_id": clip_id,
        "path": str(output_path.resolve()),
        "duration_s": round(info.duration, 2),
        "sample_rate": info.samplerate,
        "channels": info.channels,
    }


def delete_clip(project_id: str, clip_id: str) -> bool:
    """
    Delete a clip from the project's raw directory (for re-recording).

    Returns True if deleted, False if it didn't exist.
    """
    raw_dir = get_project_raw_dir(project_id)
    clip_path = raw_dir / f"{clip_id}.wav"

    if clip_path.exists():
        clip_path.unlink()
        logger.info(f"Clip deleted: project={project_id}, clip={clip_id}")
        return True

    return False


def list_clips(project_id: str) -> list[dict]:
    """
    List all raw clips for a project with basic metadata.

    Returns a list of dicts with clip_id, path, duration_s, sample_rate.
    """
    raw_dir = get_project_raw_dir(project_id)

    if not raw_dir.exists():
        return []

    clips = []
    for wav_file in sorted(raw_dir.glob("*.wav")):
        try:
            info = sf.info(str(wav_file))
            clips.append({
                "clip_id": wav_file.stem,
                "path": str(wav_file.resolve()),
                "duration_s": round(info.duration, 2),
                "sample_rate": info.samplerate,
                "channels": info.channels,
            })
        except Exception:
            # Skip corrupt files
            logger.warning(f"Skipping unreadable file: {wav_file.name}")

    return clips
