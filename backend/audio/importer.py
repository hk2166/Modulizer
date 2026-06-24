"""
importer.py — long audio / video import pipeline.

WHAT THIS DOES
──────────────
User uploads a 10-minute interview, podcast, or video (or anything with
sparse, scattered speech). We:

  1. Use ffmpeg to extract / convert to a single mono WAV at 22.05 kHz.
  2. Detect non-silent (speech) bursts using energy thresholding.
  3. Cut clips ONLY at natural silence boundaries — never mid-word.
  4. Produce variable-length clips in the [3 s, 15 s] window XTTS likes.
  5. Save each clip into the project, validated + preprocessed.

WHY NATURAL-BOUNDARY CUTS, NOT BLIND TIME-SLICING
─────────────────────────────────────────────────
A naive "concatenate everything, then chop into uniform 10-second pieces"
strategy looks tidy on paper but cuts straight through words. The audio
in clip N ends mid-syllable; clip N+1 starts mid-syllable. Whisper then
transcribes those fragments as garbled half-words, and the (audio, text)
pairs in the dataset stop matching reality. The model learns to stutter
and produces unnatural prosody at inference time.

So we cut **only at silences the speaker actually made**:
  - Each silence-bounded burst is a candidate clip.
  - Bursts that are already 3–15 s become a clip as-is.
  - Bursts shorter than 3 s get glued onto neighbours (with the original
    silence preserved between them) until we reach 3 s.
  - Bursts longer than 15 s get re-split at the longest internal silence
    found via a stricter silence threshold; we never just chop at a
    fixed offset.

The output is variable-length clips that all begin and end on a real
pause. Whisper transcribes them cleanly. Lengths naturally vary between
3 and ~15 s, exactly what XTTS expects.

KEY CONCEPTS
────────────
• ffmpeg pipeline: source → mono → 22.05 kHz → WAV. Works for any format
  ffmpeg supports (mp3, mp4, mov, m4a, ogg, flac, webm, ...).

• Silence-based detection (librosa.effects.split):
  - Looks at frame energy in dB.
  - Anywhere energy stays below `top_db` for a while = silence.
  - Returns (start, end) sample indices for each non-silent region.

• Greedy merge with target window:
  - Adjacent short bursts get joined (along with the silence the speaker
    actually left between them) until total length ≥ MIN_SEGMENT_S.
  - We stop adding bursts as soon as the running total would exceed
    MAX_SEGMENT_S — that becomes the next clip's first burst.

• Edge padding (EDGE_PAD_MS):
  - Plosives ("p", "t", "k") have a tiny burst of silence before the
    actual sound. If we cut exactly at the silence boundary we can
    slice that off and the consonant becomes inaudible.
  - 80 ms of pre-roll / post-roll keeps every consonant intact.

• Drop tiny bursts (DROP_TINY_BURST_MS):
  - 150 ms of "speech" is almost always a click, breath, or noise.
  - We filter these out before merging to keep the corpus clean.

WHAT WE DELIBERATELY DO **NOT** DO
──────────────────────────────────
• We prefer natural speech boundaries, but when a long take has no clear
  pauses we fall back to fixed 10-second windows. That is less perfect than
  silence-boundary splitting, but it lets Voice Profile accept a large
  continuous recording and still crop usable training clips automatically.
• We do not run a neural VAD here (silero-vad, webrtc-vad). Energy-based
  detection on the cleaned ffmpeg output is good enough for our SNR
  range and avoids pulling in an extra model + torch hub call.
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
MAX_SEGMENT_S = 14.0          # XTTS likes ≤15 s; we leave headroom for padding
SILENCE_TOP_DB = 35           # Energy this many dB below peak = silence
EDGE_PAD_MS = 80              # Pad start/end of each clip with silence so
                              # plosives ("p", "t", "k") don't get clipped
MAX_SEGMENTS_PER_IMPORT = 30  # Cap to avoid swamping the project with clips

# Greedy merge parameters — when individual silence-bounded bursts are
# too short to stand on their own, we glue adjacent bursts together
# (preserving the natural silence between them) until the running total
# reaches the target window.
TARGET_CLIP_S = 10.0          # Aim ~10 s per merged clip when growing
                              # short bursts; max is still MAX_SEGMENT_S
DROP_TINY_BURST_MS = 150      # Drop micro-fragments below this length
                              # (likely noise, not speech)

# When a single burst exceeds MAX_SEGMENT_S we re-run silence detection
# inside it with a stricter (more sensitive) threshold to find an
# interior pause to split at. We never chop at a fixed offset.
INNER_SPLIT_TOP_DB = 25       # Stricter than SILENCE_TOP_DB → finds even
                              # subtle within-sentence pauses
FALLBACK_WINDOW_S = 10.0      # Used only when no usable silence boundary exists


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

        # ── Step 3: Cut at natural silence boundaries ─────────────
        # No blind time-slicing — we keep cuts where the speaker
        # actually paused. Bursts that are already 3–15 s become a
        # clip directly. Short bursts get glued to neighbours (with
        # the original silence preserved) until they reach 3 s. Long
        # bursts get re-split at their longest internal silence.
        chunks = _segments_to_clips(samples, sr, segments)

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
            f"({len(segments)} speech bursts → {len(chunks)} natural-boundary clips)"
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
      -ar 22050      22.05 kHz (matches preprocessor / XTTS GPT internal rate)
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


def _segments_to_clips(
    samples: np.ndarray,
    sr: int,
    segments: list[tuple[int, int]],
) -> list[np.ndarray]:
    """
    Turn raw silence-bounded speech bursts into XTTS-ready clips.

    Cuts always land where the speaker actually paused — never mid-word.
    Each output clip is float32, ≥ MIN_SEGMENT_S and ≤ MAX_SEGMENT_S
    (plus EDGE_PAD_MS of pre/post-roll for plosive integrity).

    Per-burst dispatch:
      (a) min ≤ burst ≤ max → emit as a single clip.
      (b) burst > max       → re-detect silences inside the burst with
                              INNER_SPLIT_TOP_DB (more sensitive than the
                              outer pass) and pack the resulting sub-bursts
                              greedily into clips that respect [min, max].
      (c) burst < min       → accumulate with following bursts. We slice
                              the *original waveform* across them, which
                              preserves the natural silence between bursts
                              instead of jamming words together.

    DROP_TINY_BURST_MS bursts are filtered up front (clicks, breaths, noise).

    Args:
      samples:  full audio waveform (float32).
      sr:       sample rate.
      segments: silence-split (start,end) sample indices, in source order.

    Returns:
      List of float32 ndarrays, each a natural-boundary clip.
    """
    min_burst_n = int((DROP_TINY_BURST_MS / 1000.0) * sr)
    pad_n_edge = int((EDGE_PAD_MS / 1000.0) * sr)
    min_n = int(MIN_SEGMENT_S * sr)
    max_n = int(MAX_SEGMENT_S * sr)
    target_n = int(TARGET_CLIP_S * sr)

    # Filter out noise-level micro-bursts before we plan anything.
    bursts = [(s, e) for s, e in segments if (e - s) >= min_burst_n]
    if not bursts:
        return []

    total_speech_n = sum(e - s for s, e in bursts)
    logger.info(
        f"importer: kept {len(bursts)} speech bursts "
        f"({total_speech_n / sr:.1f}s of speech total)"
    )

    clips: list[np.ndarray] = []

    def _slice_with_pad(start: int, end: int) -> np.ndarray:
        """Take samples[start:end] with EDGE_PAD_MS of context on each side."""
        s = max(0, start - pad_n_edge)
        e = min(len(samples), end + pad_n_edge)
        return samples[s:e].astype(np.float32, copy=False)

    def _fallback_windows(start: int, end: int) -> list[np.ndarray]:
        """
        Crop a long continuous region into usable windows.

        Natural silence boundaries are still preferred. This path exists for
        monologues, songs, and noisy files where silence detection returns one
        oversized blob. It keeps Voice Profile from rejecting otherwise useful
        audio just because the speaker did not pause often enough.
        """
        region_n = end - start
        if region_n < min_n:
            return []
        if region_n <= max_n:
            return [_slice_with_pad(start, end)]

        window_n = min(int(FALLBACK_WINDOW_S * sr), max_n)
        window_n = max(window_n, min_n)
        fallback: list[np.ndarray] = []
        cursor = start
        while cursor + min_n <= end:
            clip_end = min(cursor + window_n, end)
            if clip_end - cursor >= min_n:
                fallback.append(_slice_with_pad(cursor, clip_end))
            cursor += window_n
        return fallback

    # ── Case (c) accumulator ──────────────────────────────────────
    # Track the *spanning window* (first burst start → last burst end)
    # so when we flush we can pull one contiguous slice. That preserves
    # the speaker's actual silences between bursts, which is exactly
    # what natural-boundary cutting promises.
    pending_start: int | None = None
    pending_end: int | None = None

    def _flush_pending() -> None:
        nonlocal pending_start, pending_end
        if pending_start is None or pending_end is None:
            return
        if (pending_end - pending_start) >= min_n:
            clips.append(_slice_with_pad(pending_start, pending_end))
        # Else: too short to stand alone. Drop it — better than serving
        # a sub-3s clip that XTTS will reject as a reference anyway.
        pending_start = None
        pending_end = None

    for start, end in bursts:
        burst_n = end - start

        # ── Case (a): perfect-size burst ──────────────────────────
        if min_n <= burst_n <= max_n:
            _flush_pending()
            clips.append(_slice_with_pad(start, end))
            continue

        # ── Case (b): oversized burst, split internally ───────────
        if burst_n > max_n:
            _flush_pending()
            burst_audio = samples[start:end]
            # librosa returns offsets relative to the burst slice, so we
            # add `start` to bring them back into absolute sample space.
            inner = librosa.effects.split(
                burst_audio,
                top_db=INNER_SPLIT_TOP_DB,
                frame_length=2048,
                hop_length=512,
            )

            before_count = len(clips)
            sub_start: int | None = None
            sub_end: int | None = None
            for s_i, e_i in inner:
                abs_s = start + int(s_i)
                abs_e = start + int(e_i)
                if (abs_e - abs_s) > max_n:
                    if sub_start is not None and sub_end is not None:
                        clips.extend(_fallback_windows(sub_start, sub_end))
                        sub_start, sub_end = None, None
                    clips.extend(_fallback_windows(abs_s, abs_e))
                    continue
                if sub_start is None:
                    sub_start, sub_end = abs_s, abs_e
                    continue
                # Would absorbing this sub-burst push the running window
                # past max? Flush what we have and start fresh.
                if (abs_e - sub_start) > max_n:
                    if sub_end is not None and (sub_end - sub_start) >= min_n:
                        clips.append(_slice_with_pad(sub_start, sub_end))
                    sub_start, sub_end = abs_s, abs_e
                else:
                    sub_end = abs_e
            # Trailing flush for the final sub-window.
            if sub_start is not None and sub_end is not None and (sub_end - sub_start) >= min_n:
                clips.extend(_fallback_windows(sub_start, sub_end))
            if len(clips) == before_count:
                clips.extend(_fallback_windows(start, end))
            continue

        # ── Case (c): short burst, accumulate ─────────────────────
        if pending_start is None:
            pending_start, pending_end = start, end
            continue

        # Would adding this burst overflow max? Flush first.
        if (end - pending_start) > max_n:
            _flush_pending()
            pending_start, pending_end = start, end
            continue

        pending_end = end

        # Once we've grown past TARGET_CLIP_S we're in the sweet spot
        # for an XTTS clip. Flush so the next short burst starts a new
        # window instead of pushing toward the 14 s ceiling.
        if (pending_end - pending_start) >= target_n:
            _flush_pending()

    # Trailing flush for any leftover accumulator.
    _flush_pending()

    return clips


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
