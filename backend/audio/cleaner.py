"""
cleaner.py — synthesis-grade reference clip cleaner.

WHY A SEPARATE STAGE?
─────────────────────
The general `preprocessor.py` does enough cleanup for storage and previews:
mono → resample → trim → peak-normalize. That's fine for most uses.

But XTTS v2's voice cloning is *unforgiving* of a few specific issues in
the reference clip:

  • Low-frequency rumble (HVAC, traffic, wind on the mic): pollutes the
    speaker embedding, makes the clone sound "boomy" or unstable.
  • DC offset (waveform not centered at 0): wastes dynamic range,
    can make later filters misbehave.
  • Inconsistent loudness within the clip: the model's prosody encoder
    picks up artificial level variation as "style," which leaks into
    every generation.
  • Lingering noise floor: the clone whispers/hisses in the background.

So we run a **second, synthesis-targeted pass** on the already-processed
clip and cache the result. The general processed file stays untouched
(future training will use it raw).

PIPELINE:
─────────
  processed.wav
      ↓ DC offset removal           (subtract mean)
      ↓ high-pass filter @ 80 Hz    (kills rumble, keeps voice)
      ↓ spectral denoise            (reduce steady-state noise floor)
      ↓ aggressive silence trim     (top_db=25, tighter than preprocessor)
      ↓ RMS-target normalize        (consistent perceived loudness)
      ↓ peak limiter                (prevent any post-normalize clipping)
  cleaned.wav

KEY CONCEPTS:
─────────────
• DC offset: a constant additive bias on the waveform. If your samples
  hover around 0.05 instead of 0, that's a 0.05 DC offset. Audible as
  a click on play/stop and shrinks your usable headroom.

• High-pass filter: lets high frequencies through, blocks low ones.
  We use an 80 Hz cutoff with a 4th-order Butterworth — gentle enough
  to not hollow out male voices, steep enough to nuke 50/60 Hz hum
  and HVAC rumble.

• RMS normalize vs peak normalize:
    Peak normalize sets the loudest sample to a target. One spike
      near the start dictates the level of the whole clip.
    RMS normalize sets the *average energy* to a target. Much closer
      to how humans perceive loudness. Better for synthesis.

• Limiter: a "ceiling" that prevents samples from exceeding ±1.0 after
  normalization. We use soft tanh-style clipping which is smoother
  than hard clipping.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
from scipy.signal import butter, sosfiltfilt

from backend.core.logger import logger
from backend.core.settings import DATA_DIR


# ── Configuration ─────────────────────────────────────────────────
TARGET_SAMPLE_RATE = 24000

# High-pass filter cutoff. 80 Hz keeps fundamentals of even deep male
# voices (~85 Hz) while removing rumble. Lower for very deep voices.
HPF_CUTOFF_HZ = 80.0
HPF_ORDER = 4

# Aggressive silence trimming for reference clips. The general preprocessor
# uses 30 dB; we use 25 dB here to also chop quiet breaths and lip noise
# at the edges.
TRIM_TOP_DB = 25

# RMS normalize target. -20 dBFS is a common reference for clean speech
# (broadcast standard is -23 LUFS, but RMS dBFS is simpler and close enough).
TARGET_RMS_DBFS = -20.0

# Peak limiter ceiling. Below 0 dBFS to leave headroom for any downstream
# processing inside XTTS.
LIMITER_CEILING = 0.97   # ~ -0.26 dBFS

# Spectral denoise — same simple subtraction as preprocessor but slightly
# more aggressive (use first 0.3s as noise estimate, scale subtraction up).
NOISE_DURATION_S = 0.3
NOISE_OVERSUB_FACTOR = 1.5   # Subtract 1.5× estimated noise power


@dataclass
class CleanResult:
    """Outcome of cleaning a clip for synthesis."""
    success: bool
    output_path: str | None
    input_duration_s: float
    output_duration_s: float
    rms_dbfs_before: float
    rms_dbfs_after: float
    error: str | None = None


# ══════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════


def get_project_cleaned_dir(project_id: str) -> Path:
    """data/projects/{id}/cleaned/ — created on demand."""
    cleaned_dir = DATA_DIR / "projects" / project_id / "cleaned"
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    return cleaned_dir


def clean_for_synthesis(
    input_path: str | Path,
    output_path: str | Path,
    apply_denoise: bool = True,
) -> CleanResult:
    """
    Run the synthesis-targeted cleaning pipeline.

    Args:
        input_path:    Path to a processed .wav (24 kHz mono).
        output_path:   Where to write the cleaned .wav.
        apply_denoise: Spectral denoise toggle. Off can be safer for
                       already-clean studio recordings (denoising clean
                       audio can introduce "musical noise" artifacts).

    Returns:
        CleanResult with stats. On failure, success=False and `error`
        contains a user-friendly message.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    # ── Load ──────────────────────────────────────────────────────
    try:
        samples, sr = librosa.load(str(input_path), sr=None, mono=True)
    except Exception as e:
        logger.error(f"cleaner: failed to load {input_path}: {e}")
        return CleanResult(
            success=False, output_path=None,
            input_duration_s=0.0, output_duration_s=0.0,
            rms_dbfs_before=-100.0, rms_dbfs_after=-100.0,
            error="Couldn't read the reference audio.",
        )

    input_duration_s = len(samples) / sr
    rms_before = _rms_dbfs(samples)
    logger.info(
        f"cleaner: in={input_path.name} sr={sr} "
        f"dur={input_duration_s:.2f}s rms={rms_before:.1f} dBFS"
    )

    # If the input wasn't already at target sr, resample. Cheap safety.
    if sr != TARGET_SAMPLE_RATE:
        samples = librosa.resample(samples, orig_sr=sr, target_sr=TARGET_SAMPLE_RATE)
        sr = TARGET_SAMPLE_RATE

    # ── 1. DC offset removal ──────────────────────────────────────
    # Subtract the mean of all samples. Centers the waveform on 0.
    # On very short clips this can briefly affect low frequencies, but
    # for 6+ seconds of speech it's a clear win.
    samples = samples - np.mean(samples)

    # ── 2. High-pass filter @ 80 Hz ───────────────────────────────
    # Butterworth filter is "maximally flat" in passband — no audible
    # ripple in the voice frequencies. We use SOS (second-order sections)
    # because they're numerically stable for higher orders.
    # `sosfiltfilt` applies the filter forward then backward, which gives
    # zero phase distortion (no pitch shift or smearing).
    sos = butter(HPF_ORDER, HPF_CUTOFF_HZ, btype="highpass", fs=sr, output="sos")
    samples = sosfiltfilt(sos, samples).astype(np.float32)

    # ── 3. Spectral denoise (optional) ────────────────────────────
    if apply_denoise:
        samples = _spectral_denoise(samples, sr)

    # ── 4. Aggressive silence trim ────────────────────────────────
    samples, _ = librosa.effects.trim(samples, top_db=TRIM_TOP_DB)

    if len(samples) == 0:
        logger.warning("cleaner: trim removed everything (clip was effectively silent)")
        return CleanResult(
            success=False, output_path=None,
            input_duration_s=input_duration_s, output_duration_s=0.0,
            rms_dbfs_before=rms_before, rms_dbfs_after=-100.0,
            error="The reference recording was too quiet to use.",
        )

    # ── 5. RMS-target normalize ───────────────────────────────────
    samples = _rms_normalize(samples, TARGET_RMS_DBFS)

    # ── 6. Soft limiter ───────────────────────────────────────────
    # If RMS normalize pushed any samples above the ceiling, smoothly
    # tame them with tanh-style soft clipping. Cleaner than hard clip.
    samples = _soft_limit(samples, LIMITER_CEILING)

    rms_after = _rms_dbfs(samples)
    output_duration_s = len(samples) / sr

    # ── 7. Write ──────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        sf.write(str(output_path), samples, sr, subtype="PCM_16")
    except Exception as e:
        logger.error(f"cleaner: failed to write {output_path}: {e}")
        return CleanResult(
            success=False, output_path=None,
            input_duration_s=input_duration_s, output_duration_s=0.0,
            rms_dbfs_before=rms_before, rms_dbfs_after=-100.0,
            error="Couldn't save the cleaned reference.",
        )

    logger.info(
        f"cleaner: out={output_path.name} dur={output_duration_s:.2f}s "
        f"rms_before={rms_before:.1f} → rms_after={rms_after:.1f} dBFS"
    )

    return CleanResult(
        success=True,
        output_path=str(output_path.resolve()),
        input_duration_s=round(input_duration_s, 2),
        output_duration_s=round(output_duration_s, 2),
        rms_dbfs_before=round(rms_before, 1),
        rms_dbfs_after=round(rms_after, 1),
    )


def get_or_clean_reference(project_id: str, processed_path: str | Path) -> str:
    """
    Lazy cleaner: returns a path to a synthesis-ready reference for the
    given processed clip. If a cached cleaned version exists and is newer
    than the processed source, reuses it. Otherwise re-cleans.

    Cache key: same filename, in the project's `cleaned/` directory.

    Args:
        project_id:     Project identifier (for cache location).
        processed_path: Path to the already-preprocessed clip.

    Returns:
        Absolute path to a cleaned .wav suitable for XTTS reference.

    Raises:
        ValueError: If cleaning fails (passes through the friendly error).
    """
    processed_path = Path(processed_path)
    cleaned_dir = get_project_cleaned_dir(project_id)
    cleaned_path = cleaned_dir / processed_path.name

    # Cache check: cleaned file exists AND is newer than the source
    if cleaned_path.exists():
        if cleaned_path.stat().st_mtime >= processed_path.stat().st_mtime:
            logger.info(f"cleaner: cache hit for {processed_path.name}")
            return str(cleaned_path.resolve())

    # Cache miss — produce fresh cleaned version
    result = clean_for_synthesis(processed_path, cleaned_path)
    if not result.success:
        raise ValueError(result.error or "Couldn't clean the reference audio.")
    return result.output_path  # type: ignore[return-value]


# ══════════════════════════════════════════════════════════════════
# Internal helpers
# ══════════════════════════════════════════════════════════════════


def _rms_dbfs(samples: np.ndarray) -> float:
    """RMS level in dBFS. Returns -100 for true silence."""
    rms = np.sqrt(np.mean(samples ** 2))
    if rms < 1e-10:
        return -100.0
    return 20.0 * float(np.log10(rms))


def _rms_normalize(samples: np.ndarray, target_dbfs: float) -> np.ndarray:
    """
    Scale signal so its RMS hits target_dbfs.

    HOW:
      1. current_rms = sqrt(mean(samples²))
      2. target_rms_linear = 10^(target_dbfs/20)
      3. scale = target_rms_linear / current_rms
      4. samples *= scale
    """
    rms = np.sqrt(np.mean(samples ** 2))
    if rms < 1e-10:
        return samples
    target_linear = 10 ** (target_dbfs / 20.0)
    return (samples * (target_linear / rms)).astype(np.float32)


def _soft_limit(samples: np.ndarray, ceiling: float) -> np.ndarray:
    """
    Soft limiter using tanh.

    Hard clip:  if |x| > c, set to ±c. Introduces harsh harmonics.
    Soft clip:  c * tanh(x / c). Smoothly rolls off as |x| → ∞,
                only audible distortion when significantly over.

    For typical speech this leaves most samples untouched; only the
    occasional plosive peak gets gently rounded.
    """
    return (ceiling * np.tanh(samples / ceiling)).astype(np.float32)


def _spectral_denoise(samples: np.ndarray, sr: int) -> np.ndarray:
    """
    Spectral subtraction with oversubtraction.

    Differs from preprocessor's version:
      - Slightly shorter noise window (0.3s vs 0.5s) — most clips have
        less leading silence after the general preprocess.
      - Oversubtraction factor 1.5 — more aggressive cleanup. Tradeoff:
        risk of "musical noise" artifacts, but for synthesis reference
        a slightly drier sound is preferable to residual hiss.
    """
    noise_n = int(NOISE_DURATION_S * sr)
    if len(samples) <= noise_n + sr:  # need >1s of audio after noise window
        return samples

    n_fft = 2048
    hop = 512

    stft = librosa.stft(samples, n_fft=n_fft, hop_length=hop)
    noise_stft = librosa.stft(samples[:noise_n], n_fft=n_fft, hop_length=hop)
    noise_power = np.mean(np.abs(noise_stft) ** 2, axis=1, keepdims=True)

    signal_power = np.abs(stft) ** 2
    clean_power = np.maximum(signal_power - NOISE_OVERSUB_FACTOR * noise_power, 0)

    clean_mag = np.sqrt(clean_power)
    phase = np.angle(stft)
    clean_stft = clean_mag * np.exp(1j * phase)

    return librosa.istft(
        clean_stft, hop_length=hop, length=len(samples)
    ).astype(np.float32)
