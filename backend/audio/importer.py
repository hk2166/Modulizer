"""
importer.py — long audio / video import pipeline.

WHAT THIS DOES
──────────────
User uploads a 10-minute interview, podcast, or video (or anything with
sparse, scattered speech). We:

  1. Use ffmpeg to extract / convert to a single mono WAV at 24 kHz.
  2. Detect non-silent (speech) bursts using energy thresholding.
  3. **Concatenate** all the speech together with tiny gaps between bursts.
  4. Slice the resulting continuous track into uniform ~10-second clips.
  5. Save each clip into the project, validated + preprocessed.

WHY CONCATENATE-THEN-CHUNK INSTEAD OF FILTER-BY-LENGTH
───────────────────────────────────────────────────────
Naive silence-splitting + length filter throws away every burst under 3s.
A recording with scattered speech (a few words → silence → a sentence →
silence → ...) would produce zero usable clips even though it might have
several minutes of usable speech total.

Concatenating first lets us salvage every bit of audio. Chunking afterward
gives us uniform clip sizes XTTS likes. Works for both:
  - Continuous speech (podcasts, monologues): bursts are long, gaps short.
    Chunks just slice the natural flow into 10-second pieces.
  - Scattered speech (voicemails, conversational recordings): bursts are
    short, gaps long. We glue them together and ship 10-second compilations.

KEY CONCEPTS
────────────
• ffmpeg pipeline: source → mono → 24 kHz → WAV float32. Works for any
  format ffmpeg supports (mp3, mp4, mov, m4a, ogg, flac, webm, ...).

• Silence-based detection (librosa.effects.split):
  - Looks at frame energy in dB.
  - Anywhere energy stays below `top_db` for a while = silence.
  - Returns (start, end) sample indices for each non-silent region.

• Concatenation gap (INTRA_CLIP_GAP_MS):
  - We don't smash bursts directly together; that can audibly merge
    the last word of one burst with the first of the next ("hello"
    + "world" → "helloworld").
  - 200ms of silence between bursts feels natural and matches how
    speakers actually pause.

• Drop tiny bursts (DROP_TINY_BURST_MS):
  - 50ms of "speech" is almost always a click, breath, or noise.
  - We filter these out before concatenating to keep the corpus clean.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import librosa
import numpy as np
import soundfile as sf
import imageio_ffmpeg

from backend.audio.recorder import save_clip, RecorderError
from backend.audio.preprocessor import preprocess_clip, TARGET_SAMPLE_RATE
from backend.audio.validator import validate_clip
from backend.core.logger import logger
from backend.core.settings import DATA_DIR


# ── Configuration ─────────────────────────────────────────────────
MIN_SEGMENT_S = 3.0           # Below this is too short for XTTS reference
MAX_SEGMENT_S = 14.0          # XTTS likes ≤15s; we leave headroom for padding
SILENCE_TOP_DB = 35           # Energy this many dB below peak = silence
EDGE_PAD_MS = 80              # Pad start/end of each segment with silence so
                              # plosives ("p", "t", "k") don't get clipped
MAX_SEGMENTS_PER_IMPORT = 30  # Cap to avoid swamping the project with tiny clips

# Concatenation strategy — for recordings with scattered speech (lots of
# short bursts separated by silence), we glue all speech together and then
# slice it into uniform target-length clips. See _concat_and_chunk.
TARGET_CLIP_S = 10.0          # Aim for ~10s per generated clip
INTRA_CLIP_GAP_MS = 200       # Tiny silence between glued bursts so words
                              # don't run into each other unnaturally
DROP_TINY_BURST_MS = 150      # Drop micro-fragments below this length
                              # (likely noise, not speech)


@dataclass
class ImportResult:
    """Outcome of importing a single source file."""
    success: bool
    source_filename: str
    source_duration_s: float
    segments_found: int            # before filtering
    segments_kept: int             # after duration filter + cap
    clip_ids: list[str]            # IDs of clips saved into the project
    error: str | None = None


# ══════════════════════════════════════════════════════════════════
# Public entry points
# ══════════════════════════════════════════════════════════════════


def import_recording(project_id: str, source_path: str | Path) -> ImportResult:
    """
    Top-level: take any audio/video file and turn it into clips on the project.

    Steps (each can raise; we catch and return ImportResult.success=False):
      1. Extract a normalized mono WAV via ffmpeg
      2. Split that WAV into speech segments
      3. For each segment, save → validate → preprocess just like an upload
      4. Clean up the temp extracted WAV

    Args:
        project_id: project to add clips to.
        source_path: any file ffmpeg can read (.mp4, .mp3, .m4a, .wav, .mov...)

    Returns:
        ImportResult — segment counts, clip ids, or friendly error.
    """
    source_path = Path(source_path)
    logger.info(f"importer: starting import of {source_path.name}")

    # ── Step 1: Extract mono 24 kHz WAV via ffmpeg ────────────────
    extracted_wav = DATA_DIR / "temp" / f"import_{uuid4()}.wav"
    extracted_wav.parent.mkdir(parents=True, exist_ok=True)

    try:
        _ffmpeg_extract_audio(source_path, extracted_wav)
    except FFmpegError as e:
        return ImportResult(
            success=False,
            source_filename=source_path.name,
            source_duration_s=0.0,
            segments_found=0,
            segments_kept=0,
            clip_ids=[],
            error=str(e),
        )

    try:
        # ── Step 2: Split into speech segments ────────────────────
        try:
            samples, sr = sf.read(str(extracted_wav), dtype="float32")
        except Exception as e:
            return ImportResult(
                success=False,
                source_filename=source_path.name,
                source_duration_s=0.0,
                segments_found=0,
                segments_kept=0,
                clip_ids=[],
                error=f"Couldn't read the extracted audio: {e}",
            )

        source_duration_s = len(samples) / sr
        logger.info(
            f"importer: extracted {source_duration_s:.1f}s of audio at {sr} Hz"
        )

        segments = _split_into_segments(samples, sr)
        if not segments:
            return ImportResult(
                success=False,
                source_filename=source_path.name,
                source_duration_s=source_duration_s,
                segments_found=0,
                segments_kept=0,
                clip_ids=[],
                error="Couldn't find any clear speech in this recording.",
            )

        # ── Step 3: Concatenate all the speech, then re-chunk ─────
        # This handles two cases uniformly:
        #   a) Continuous speech: most segments are already long enough.
        #   b) Spotty speech (your case): individual bursts are too short,
        #      but glued together they make plenty of usable clips.
        #
        # We end up with uniform ~10s clips regardless of source rhythm.
        chunks = _concat_and_chunk(samples, sr, segments)

        if not chunks:
            return ImportResult(
                success=False,
                source_filename=source_path.name,
                source_duration_s=source_duration_s,
                segments_found=len(segments),
                segments_kept=0,
                clip_ids=[],
                error=(
                    f"Found {len(segments)} speech bursts but couldn't make "
                    f"any usable clips out of them. The recording might be "
                    f"too quiet or too short."
                ),
            )

        # Cap to MAX_SEGMENTS_PER_IMPORT (keep first N — they're in source order)
        chunks = chunks[:MAX_SEGMENTS_PER_IMPORT]

        # ── Step 4: Save → validate → preprocess each chunk ───────
        clip_ids = _save_chunks(project_id, chunks, sr)

        logger.info(
            f"importer: imported {len(clip_ids)} clips from {source_path.name} "
            f"({len(segments)} speech bursts → {len(chunks)} ~{TARGET_CLIP_S:.0f}s clips)"
        )

        return ImportResult(
            success=True,
            source_filename=source_path.name,
            source_duration_s=round(source_duration_s, 2),
            segments_found=len(segments),
            segments_kept=len(clip_ids),
            clip_ids=clip_ids,
        )

    finally:
        # Clean up the temp extracted WAV regardless of outcome
        try:
            extracted_wav.unlink(missing_ok=True)
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════
# Internals
# ══════════════════════════════════════════════════════════════════


class FFmpegError(Exception):
    """Raised when ffmpeg fails to extract audio."""


def _ffmpeg_extract_audio(source: Path, output_wav: Path) -> None:
    """
    Run ffmpeg: source → mono 24 kHz signed-16 WAV.

    Args:
      source: any video/audio file ffmpeg can read
      output_wav: where to write the extracted WAV

    Flags:
      -y             overwrite output if it exists
      -i source      input file
      -vn            drop video stream (we only want audio)
      -ac 1          mono
      -ar 24000      24 kHz (matches XTTS native rate; saves a resample later)
      -acodec pcm_s16le  16-bit signed little-endian PCM (standard WAV)
      -loglevel error    only show actual errors, not progress noise
    """
    ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()

    cmd = [
        ffmpeg_bin,
        "-y",
        "-i", str(source),
        "-vn",
        "-ac", "1",
        "-ar", str(TARGET_SAMPLE_RATE),
        "-acodec", "pcm_s16le",
        "-loglevel", "error",
        str(output_wav),
    ]

    logger.info(f"ffmpeg: {' '.join(cmd[1:])}")  # skip binary path in log

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 min — generous for big files on slow disks
        )
    except subprocess.TimeoutExpired:
        raise FFmpegError("That recording took too long to process. Try a shorter file.")
    except FileNotFoundError:
        raise FFmpegError("Audio extractor isn't available. Reinstall the app.")

    if result.returncode != 0:
        # Friendly message; technical detail in logs
        logger.error(f"ffmpeg failed: {result.stderr}")
        raise FFmpegError(
            "We couldn't read that file. Make sure it's a real audio or video recording."
        )


def _split_into_segments(samples: np.ndarray, sr: int) -> list[tuple[int, int]]:
    """
    Use silence detection to find non-silent (speech) regions.

    Returns list of (start_sample, end_sample) pairs, in source order.
    librosa.effects.split returns these as a (N, 2) ndarray.
    """
    # frame_length and hop_length control how finely we look at energy.
    # Defaults are good for speech; smaller hop = finer detection but slower.
    intervals = librosa.effects.split(
        samples,
        top_db=SILENCE_TOP_DB,
        frame_length=2048,
        hop_length=512,
    )
    return [(int(s), int(e)) for s, e in intervals]


def _concat_and_chunk(
    samples: np.ndarray,
    sr: int,
    segments: list[tuple[int, int]],
) -> list[np.ndarray]:
    """
    Glue all speech segments together, then slice into ~TARGET_CLIP_S clips.

    Why this approach:
      Naive silence-splitting throws away short bursts. A recording with
      lots of brief speech ("Hi" — silence — "How are you" — silence — ...)
      ends up producing zero usable clips because each piece is under the
      3s minimum.

      Concatenating first means we can salvage every bit of speech, then
      decide how to package it.

    Steps:
      1. Drop ultra-short bursts (< DROP_TINY_BURST_MS). Likely noise.
      2. Insert a tiny silence between bursts so words don't smush together.
         This silence is too short to hurt cloning but long enough to avoid
         "gluing two words into one weird hybrid."
      3. Concatenate everything into one continuous speech track.
      4. Chunk the track into TARGET_CLIP_S windows.
      5. The last chunk may be short — keep it if ≥ MIN_SEGMENT_S, else drop.

    Args:
      samples:  full audio waveform.
      sr:       sample rate.
      segments: silence-split (start,end) sample indices.

    Returns:
      List of float32 ndarrays, each ~TARGET_CLIP_S long.
    """
    min_burst_n = int((DROP_TINY_BURST_MS / 1000.0) * sr)
    pad_n_edge = int((EDGE_PAD_MS / 1000.0) * sr)
    gap_n = int((INTRA_CLIP_GAP_MS / 1000.0) * sr)
    target_n = int(TARGET_CLIP_S * sr)
    min_keep_n = int(MIN_SEGMENT_S * sr)

    # ── Step 1: extract speech bursts above the noise threshold ───
    bursts: list[np.ndarray] = []
    for start, end in segments:
        if end - start < min_burst_n:
            continue
        # Add a small edge pad so plosives at burst boundaries are intact.
        s = max(0, start - pad_n_edge)
        e = min(len(samples), end + pad_n_edge)
        bursts.append(samples[s:e])

    if not bursts:
        return []

    total_speech_n = sum(len(b) for b in bursts)
    logger.info(
        f"importer: kept {len(bursts)} speech bursts "
        f"({total_speech_n / sr:.1f}s of speech total)"
    )

    # If we don't even have one full clip's worth of speech, bail out
    # rather than producing a weirdly short single output.
    if total_speech_n < min_keep_n:
        return []

    # ── Step 2 & 3: glue with small silence gaps ──────────────────
    silence = np.zeros(gap_n, dtype=np.float32)
    glued_parts: list[np.ndarray] = []
    for i, b in enumerate(bursts):
        if i > 0:
            glued_parts.append(silence)
        glued_parts.append(b.astype(np.float32))
    glued = np.concatenate(glued_parts)

    # ── Step 4: slice into TARGET_CLIP_S chunks ───────────────────
    chunks: list[np.ndarray] = []
    cursor = 0
    while cursor + target_n <= len(glued):
        chunks.append(glued[cursor:cursor + target_n])
        cursor += target_n

    # ── Step 5: handle tail ───────────────────────────────────────
    tail = glued[cursor:]
    if len(tail) >= min_keep_n:
        chunks.append(tail)
    elif chunks:
        # If tail is too short to stand alone but we have at least one chunk,
        # we drop the tail rather than padding the last clip past the target.
        # Better to lose 2s of speech than serve a Frankenstein 12s clip.
        logger.info(
            f"importer: dropping {len(tail)/sr:.1f}s tail (below {MIN_SEGMENT_S}s)"
        )

    return chunks


def _save_chunks(
    project_id: str,
    chunks: list[np.ndarray],
    sr: int,
) -> list[str]:
    """
    Save each chunk as a project clip and run validate + preprocess.

    Same pipeline as a normal upload, just driven from in-memory arrays.
    Returns list of clip IDs that successfully landed in the project.
    """
    saved_clip_ids: list[str] = []

    for i, chunk in enumerate(chunks):
        try:
            wav_bytes = _samples_to_wav_bytes(chunk, sr)
        except Exception as e:
            logger.warning(f"importer: encode failed for chunk {i}: {e}")
            continue

        try:
            clip = save_clip(project_id, wav_bytes, filename=f"import_{i:02d}.wav")
        except RecorderError as e:
            logger.warning(f"importer: save failed for chunk {i}: {e}")
            continue

        # Validate (informational only — these came from a longer source,
        # so we keep them even if a check fails)
        validation = validate_clip(clip["path"])
        if not validation.valid:
            logger.info(
                f"importer: chunk {i} validation: {validation.errors} "
                f"(keeping anyway)"
            )

        result = preprocess_clip(
            project_id=project_id,
            clip_id=clip["clip_id"],
            input_path=clip["path"],
        )
        if not result.success:
            logger.warning(f"importer: preprocess failed for chunk {i}: {result.error}")
            continue

        saved_clip_ids.append(clip["clip_id"])

    return saved_clip_ids


def _samples_to_wav_bytes(samples: np.ndarray, sr: int) -> bytes:
    """
    Encode float32 samples to in-memory WAV bytes (PCM_16).

    Why in-memory? `save_clip` accepts bytes, mirroring the HTTP upload
    path. Keeps the import pipeline using the same code paths as a normal
    file upload — fewer surprises, fewer bugs.
    """
    import io
    buf = io.BytesIO()
    sf.write(buf, samples, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()
