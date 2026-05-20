"""
validator.py — Per-clip audio quality checks with user-friendly feedback.

HOW IT WORKS:
─────────────
Each check is a small function that returns either None (pass) or a
user-friendly error string (fail). We run all checks and collect results.

CHECKS PERFORMED:
  1. Duration    — Must be 3–15 seconds (too short = not enough data,
                   too long = harder to process cleanly)
  2. SNR         — Signal-to-noise ratio must be above threshold.
                   Measures how much "voice" vs "background hiss" there is.
  3. Clipping    — Peak amplitude must be below -1 dBFS.
                   Clipping = distortion when the mic input is too loud.
  4. Sample rate — Must be ≥ 24 kHz (XTTS target). Lower rates lose
                   high-frequency detail that makes voices sound natural.

KEY CONCEPTS:
─────────────
• dBFS (decibels relative to full scale):
  - 0 dBFS = maximum possible amplitude in digital audio
  - -1 dBFS = just below max (our clipping threshold)
  - -60 dBFS = very quiet (near silence)
  Formula: dBFS = 20 * log10(amplitude), where amplitude is 0.0–1.0

• SNR (signal-to-noise ratio):
  - Ratio of "loud parts" (speech) to "quiet parts" (noise floor)
  - Higher = cleaner recording. We want ≥ 20 dB.
  - Measured by comparing RMS of the full signal vs silent segments.

• RMS (root mean square):
  - A way to measure the "average loudness" of a signal.
  - sqrt(mean(samples²)) — gives a single number for overall level.

• Sample rate:
  - How many audio samples per second (Hz).
  - 24000 Hz = 24 kHz, the target for XTTS v2.
  - Higher is fine (we'll resample later), lower loses quality.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from backend.core.logger import logger


# ── Configuration ─────────────────────────────────────────────────
MIN_DURATION_S = 3.0        # Minimum clip length in seconds
MAX_DURATION_S = 15.0       # Maximum clip length in seconds
MIN_SAMPLE_RATE = 24000     # XTTS v2 target sample rate
MIN_SNR_DB = 20.0           # Minimum signal-to-noise ratio (dB)
CLIPPING_THRESHOLD_DBFS = -1.0  # Peak must be below this (dBFS)
SILENCE_THRESHOLD_DBFS = -40.0  # Anything below this is "silence/noise"
MAX_SILENCE_RATIO = 0.7    # Max fraction of clip that can be silence


@dataclass
class ValidationResult:
    """Result of validating a single audio clip."""
    valid: bool
    errors: list[str]       # User-friendly error messages
    warnings: list[str]     # Non-blocking suggestions
    metadata: dict          # Technical details (for logging, not shown to user)


def validate_clip(file_path: str | Path) -> ValidationResult:
    """
    Run all quality checks on an audio clip.

    Args:
        file_path: Path to the .wav file to validate.

    Returns:
        ValidationResult with pass/fail status and friendly messages.
    """
    file_path = Path(file_path)
    errors: list[str] = []
    warnings: list[str] = []
    metadata: dict = {}

    # ── Load the audio ────────────────────────────────────────────
    try:
        # sf.read returns (samples_array, sample_rate)
        # samples are normalized to [-1.0, 1.0] float range
        samples, sample_rate = sf.read(str(file_path), dtype="float32")
    except Exception as e:
        logger.error(f"Failed to read audio file {file_path}: {e}")
        return ValidationResult(
            valid=False,
            errors=["This file couldn't be read. Please try recording again."],
            warnings=[],
            metadata={"error": str(e)},
        )

    # If stereo, convert to mono by averaging channels
    # Shape: (num_samples,) for mono, (num_samples, channels) for multi
    if samples.ndim > 1:
        samples = np.mean(samples, axis=1)

    duration_s = len(samples) / sample_rate
    metadata["duration_s"] = round(duration_s, 2)
    metadata["sample_rate"] = sample_rate
    metadata["num_samples"] = len(samples)

    # ── Check 1: Duration ─────────────────────────────────────────
    duration_error = _check_duration(duration_s)
    if duration_error:
        errors.append(duration_error)

    # ── Check 2: Sample rate ──────────────────────────────────────
    sr_error = _check_sample_rate(sample_rate)
    if sr_error:
        # This is a warning, not a hard error — we can resample later
        warnings.append(sr_error)
    metadata["sample_rate_ok"] = sr_error is None

    # ── Check 3: Clipping ─────────────────────────────────────────
    clipping_error = _check_clipping(samples)
    if clipping_error:
        errors.append(clipping_error)
    metadata["peak_dbfs"] = round(_amplitude_to_dbfs(np.max(np.abs(samples))), 1)

    # ── Check 4: SNR / silence ratio ─────────────────────────────
    snr_error = _check_snr(samples, sample_rate)
    if snr_error:
        errors.append(snr_error)

    # ── Build result ──────────────────────────────────────────────
    result = ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        metadata=metadata,
    )

    if result.valid:
        logger.info(f"Clip validated OK: {file_path.name}")
    else:
        logger.info(f"Clip failed validation: {file_path.name} — {errors}")

    return result


# ══════════════════════════════════════════════════════════════════
# Individual check functions
# Each returns None (pass) or a user-friendly error string (fail).
# ══════════════════════════════════════════════════════════════════


def _check_duration(duration_s: float) -> str | None:
    """
    Check if clip duration is within acceptable range.

    WHY: Too short = not enough speech data for the model to learn from.
         Too long = harder to read cleanly, more chance of mistakes.
    """
    if duration_s < MIN_DURATION_S:
        return (
            f"That was a bit short ({duration_s:.1f}s). "
            f"Try to speak for at least {MIN_DURATION_S:.0f} seconds."
        )
    if duration_s > MAX_DURATION_S:
        return (
            f"That was a bit long ({duration_s:.1f}s). "
            f"Try to keep it under {MAX_DURATION_S:.0f} seconds."
        )
    return None


def _check_sample_rate(sample_rate: int) -> str | None:
    """
    Check if sample rate is high enough for XTTS.

    WHY: XTTS v2 works at 24 kHz. If the source is lower, resampling
         can't recover the lost high frequencies — voice sounds muffled.
         Higher rates (44.1k, 48k) are fine; we downsample later.
    """
    if sample_rate < MIN_SAMPLE_RATE:
        return (
            f"Your microphone is recording at {sample_rate} Hz, which is a bit low. "
            f"If possible, set it to at least 24,000 Hz in your system audio settings."
        )
    return None


def _check_clipping(samples: np.ndarray) -> str | None:
    """
    Check if the audio is clipping (too loud, causing distortion).

    HOW: Look at the peak amplitude. If it's at or above -1 dBFS,
         the signal is hitting the digital ceiling and distorting.

    WHY: Clipped audio has flat-topped waveforms instead of smooth peaks.
         The model can't learn natural voice characteristics from distorted audio.
    """
    peak_amplitude = np.max(np.abs(samples))
    peak_dbfs = _amplitude_to_dbfs(peak_amplitude)

    if peak_dbfs >= CLIPPING_THRESHOLD_DBFS:
        return (
            "The recording is too loud and sounds distorted. "
            "Try moving a bit further from the microphone, or lower your input volume."
        )
    return None


def _check_snr(samples: np.ndarray, sample_rate: int) -> str | None:
    """
    Check signal-to-noise ratio and silence ratio.

    HOW THIS WORKS:
    1. Split the audio into small frames (50ms each).
    2. Calculate RMS energy of each frame.
    3. Frames below SILENCE_THRESHOLD_DBFS are "silent/noise".
    4. Frames above are "signal" (speech).
    5. SNR = 20 * log10(rms_signal / rms_noise)
    6. Also check that silence doesn't dominate the clip.

    WHY: If the room is noisy (fan, traffic, etc.), the model learns
         the noise as part of your voice. Clean recordings = better clone.
    """
    # Split into 50ms frames
    frame_size = int(sample_rate * 0.05)  # 50ms worth of samples
    num_frames = len(samples) // frame_size

    if num_frames < 2:
        # Clip too short to analyze meaningfully
        return None

    # Calculate RMS for each frame
    # RMS = sqrt(mean(samples²)) — measures average energy
    frames = samples[:num_frames * frame_size].reshape(num_frames, frame_size)
    frame_rms = np.sqrt(np.mean(frames ** 2, axis=1))

    # Convert to dBFS for thresholding
    frame_dbfs = np.where(
        frame_rms > 0,
        20 * np.log10(np.maximum(frame_rms, 1e-10)),
        -100.0  # Treat true silence as -100 dBFS
    )

    # Classify frames as signal or noise
    signal_mask = frame_dbfs > SILENCE_THRESHOLD_DBFS
    noise_mask = ~signal_mask

    signal_count = np.sum(signal_mask)
    noise_count = np.sum(noise_mask)

    # ── Check silence ratio ───────────────────────────────────────
    silence_ratio = noise_count / num_frames
    if silence_ratio > MAX_SILENCE_RATIO:
        return (
            "Most of that recording was silence. "
            "Make sure you're speaking clearly throughout the clip."
        )

    # ── Check SNR ─────────────────────────────────────────────────
    if signal_count == 0:
        return "We couldn't detect any speech. Please try recording again."

    if noise_count == 0:
        # No noise frames detected — SNR is effectively infinite (great!)
        return None

    # RMS of signal frames vs noise frames
    signal_rms = np.sqrt(np.mean(frames[signal_mask] ** 2))
    noise_rms = np.sqrt(np.mean(frames[noise_mask] ** 2))

    if noise_rms <= 0:
        return None  # No measurable noise — perfect

    snr_db = 20 * np.log10(signal_rms / noise_rms)

    if snr_db < MIN_SNR_DB:
        return (
            "There's quite a bit of background noise in that recording. "
            "Try finding a quieter spot, or move closer to the microphone."
        )

    return None


# ══════════════════════════════════════════════════════════════════
# Utility functions
# ══════════════════════════════════════════════════════════════════


def _amplitude_to_dbfs(amplitude: float) -> float:
    """
    Convert a linear amplitude (0.0–1.0) to dBFS.

    dBFS = 20 * log10(amplitude)

    Examples:
        1.0   →   0.0 dBFS  (maximum)
        0.5   →  -6.0 dBFS
        0.1   → -20.0 dBFS
        0.001 → -60.0 dBFS  (very quiet)
    """
    if amplitude <= 0:
        return -100.0  # Effectively silent
    return 20 * np.log10(amplitude)
