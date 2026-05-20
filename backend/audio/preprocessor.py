"""
preprocessor.py — FFmpeg-style audio pipeline using librosa + soundfile.

PIPELINE (in order):
────────────────────
  raw .wav  →  load  →  mono  →  resample  →  trim silence  →  normalize  →  processed .wav

WHY THIS ORDER MATTERS:
  - Mono first: cheaper to resample one channel than two
  - Resample before trim: trim uses energy thresholds that depend on sample rate
  - Trim before normalize: don't normalize silence padding into the signal
  - Normalize last: final loudness target applied to clean, trimmed audio

KEY CONCEPTS:
─────────────
• Resampling:
  Converting between sample rates. Going from 44.1 kHz → 24 kHz means
  we keep every 24000/44100 ≈ 0.54th sample (with interpolation).
  librosa uses a high-quality sinc filter for this.

• Silence trimming:
  librosa.effects.trim() looks at the short-time energy of the signal.
  Any frames at the start/end below a threshold (in dB) get cut.
  Think of it as "find where the voice actually starts and ends."

• Peak normalization:
  Scale the entire signal so the loudest sample hits a target dBFS.
  Formula: scale_factor = 10^(target_dBFS / 20) / peak_amplitude
  This is simpler than EBU R128 (which measures perceived loudness)
  but good enough for training data where consistency matters more
  than broadcast compliance.

• Optional denoise:
  A simple spectral subtraction approach — estimate the noise floor
  from the first 0.5s (assumed to be silence/room tone), then subtract
  that spectrum from the whole signal. Not perfect, but removes
  steady-state noise like fan hum or AC.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa

from backend.core.logger import logger
from backend.core.settings import DATA_DIR


# ── Configuration ─────────────────────────────────────────────────
TARGET_SAMPLE_RATE = 24000          # XTTS v2 native sample rate
TARGET_PEAK_DBFS = -3.0             # Normalize peaks to -3 dBFS
TRIM_TOP_DB = 30                    # Silence threshold for trimming (dB below peak)
                                    # Higher = more aggressive trimming
DENOISE_NOISE_DURATION_S = 0.5      # Use first 0.5s to estimate noise floor


@dataclass
class PreprocessResult:
    """Result of preprocessing a single clip."""
    success: bool
    output_path: str | None         # Path to processed file
    input_duration_s: float         # Duration before processing
    output_duration_s: float        # Duration after silence trim
    sample_rate: int                # Always TARGET_SAMPLE_RATE on success
    error: str | None = None        # User-friendly error if failed


def get_project_processed_dir(project_id: str) -> Path:
    """
    Get (and create) the processed audio directory for a project.

    Layout:
        data/projects/{project_id}/processed/
    """
    processed_dir = DATA_DIR / "projects" / project_id / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    return processed_dir


def preprocess_clip(
    project_id: str,
    clip_id: str,
    input_path: str | Path,
    denoise: bool = False,
) -> PreprocessResult:
    """
    Run the full preprocessing pipeline on a single clip.

    Args:
        project_id:  Project this clip belongs to.
        clip_id:     Clip identifier (used as output filename).
        input_path:  Path to the raw .wav file.
        denoise:     Whether to apply light spectral denoising.

    Returns:
        PreprocessResult with output path and metadata.
    """
    input_path = Path(input_path)

    # ── Step 1: Load audio ────────────────────────────────────────
    # librosa.load() always returns float32 samples in [-1.0, 1.0]
    # mono=True collapses stereo to mono by averaging channels
    # sr=None means "keep the original sample rate for now"
    try:
        samples, original_sr = librosa.load(str(input_path), sr=None, mono=True)
    except Exception as e:
        logger.error(f"Failed to load {input_path}: {e}")
        return PreprocessResult(
            success=False,
            output_path=None,
            input_duration_s=0.0,
            output_duration_s=0.0,
            sample_rate=0,
            error="Couldn't read that recording. Please try again.",
        )

    input_duration_s = len(samples) / original_sr
    logger.info(
        f"Preprocessing clip={clip_id}: "
        f"sr={original_sr}, duration={input_duration_s:.2f}s"
    )

    # ── Step 2: Resample to 24 kHz ────────────────────────────────
    # librosa.resample uses a polyphase filter — high quality, no aliasing.
    # We skip this if already at target rate (no-op would waste time).
    if original_sr != TARGET_SAMPLE_RATE:
        samples = librosa.resample(
            samples,
            orig_sr=original_sr,
            target_sr=TARGET_SAMPLE_RATE,
        )
        logger.info(f"Resampled {original_sr} Hz → {TARGET_SAMPLE_RATE} Hz")

    # ── Step 3: Trim leading/trailing silence ─────────────────────
    # librosa.effects.trim() returns (trimmed_samples, (start_sample, end_sample))
    # top_db: frames more than top_db dB below the peak are considered silence.
    # A value of 30 means: if peak is at -3 dBFS, silence threshold is -33 dBFS.
    samples, trim_indices = librosa.effects.trim(samples, top_db=TRIM_TOP_DB)
    trimmed_start_s = trim_indices[0] / TARGET_SAMPLE_RATE
    trimmed_end_s = trim_indices[1] / TARGET_SAMPLE_RATE
    logger.info(
        f"Trimmed silence: kept [{trimmed_start_s:.2f}s – {trimmed_end_s:.2f}s]"
    )

    # ── Step 4: Optional denoise ──────────────────────────────────
    if denoise:
        samples = _spectral_denoise(samples, TARGET_SAMPLE_RATE)
        logger.info("Applied spectral denoising")

    # ── Step 5: Peak normalize ────────────────────────────────────
    samples = _peak_normalize(samples, TARGET_PEAK_DBFS)

    # ── Step 6: Write output ──────────────────────────────────────
    processed_dir = get_project_processed_dir(project_id)
    output_path = processed_dir / f"{clip_id}.wav"

    try:
        sf.write(str(output_path), samples, TARGET_SAMPLE_RATE, subtype="PCM_16")
    except Exception as e:
        logger.error(f"Failed to write processed clip: {e}")
        return PreprocessResult(
            success=False,
            output_path=None,
            input_duration_s=input_duration_s,
            output_duration_s=0.0,
            sample_rate=TARGET_SAMPLE_RATE,
            error="Couldn't save the processed recording.",
        )

    output_duration_s = len(samples) / TARGET_SAMPLE_RATE
    logger.info(
        f"Preprocessing done: {output_path.name}, "
        f"duration={output_duration_s:.2f}s"
    )

    return PreprocessResult(
        success=True,
        output_path=str(output_path.resolve()),
        input_duration_s=round(input_duration_s, 2),
        output_duration_s=round(output_duration_s, 2),
        sample_rate=TARGET_SAMPLE_RATE,
    )


def preprocess_project(
    project_id: str,
    clip_map: dict[str, str],
    denoise: bool = False,
) -> dict[str, PreprocessResult]:
    """
    Preprocess all clips for a project.

    Args:
        project_id: Project identifier.
        clip_map:   Dict of {clip_id: raw_file_path}.
        denoise:    Whether to apply denoising to all clips.

    Returns:
        Dict of {clip_id: PreprocessResult}.
    """
    results = {}
    for clip_id, raw_path in clip_map.items():
        results[clip_id] = preprocess_clip(project_id, clip_id, raw_path, denoise)
    return results


# ══════════════════════════════════════════════════════════════════
# Internal helpers
# ══════════════════════════════════════════════════════════════════


def _peak_normalize(samples: np.ndarray, target_dbfs: float) -> np.ndarray:
    """
    Scale audio so the peak amplitude hits target_dbfs.

    HOW:
      1. Find the current peak (max absolute value).
      2. Convert target_dbfs to a linear amplitude: target_amp = 10^(dBFS/20)
      3. Scale factor = target_amp / current_peak
      4. Multiply all samples by scale factor.

    Example:
      Current peak = 0.5 (-6 dBFS), target = -3 dBFS (0.708 linear)
      Scale factor = 0.708 / 0.5 = 1.416
      All samples multiplied by 1.416 → peak now at 0.708 = -3 dBFS
    """
    peak = np.max(np.abs(samples))
    if peak < 1e-10:
        # Silent clip — nothing to normalize
        return samples

    target_amplitude = 10 ** (target_dbfs / 20.0)
    scale_factor = target_amplitude / peak
    return samples * scale_factor


def _spectral_denoise(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Light spectral subtraction denoising.

    HOW IT WORKS:
      1. Take the first DENOISE_NOISE_DURATION_S seconds as a "noise sample"
         (assumes the recording starts with a moment of silence/room tone).
      2. Compute the average power spectrum of that noise sample using STFT.
         STFT = Short-Time Fourier Transform: breaks audio into overlapping
         frames and computes frequency content of each frame.
      3. Subtract the noise spectrum from every frame of the full signal.
         Frequencies dominated by noise get attenuated.
      4. Reconstruct audio from the modified spectrum using inverse STFT.

    LIMITATIONS:
      - Only removes steady-state noise (fan hum, AC, white noise).
      - Won't help with intermittent noise (coughs, keyboard clicks).
      - Can introduce "musical noise" artifacts if noise estimate is wrong.
      - That's why it's optional.
    """
    noise_samples = int(DENOISE_NOISE_DURATION_S * sample_rate)

    if len(samples) <= noise_samples:
        # Clip too short to estimate noise — skip
        return samples

    # STFT parameters
    n_fft = 2048        # FFT window size (frequency resolution)
    hop_length = 512    # Step between frames (time resolution)

    # Compute STFT of full signal → complex matrix (freq_bins × time_frames)
    stft = librosa.stft(samples, n_fft=n_fft, hop_length=hop_length)

    # Estimate noise from first N samples
    noise_stft = librosa.stft(
        samples[:noise_samples], n_fft=n_fft, hop_length=hop_length
    )
    # Mean power spectrum of noise (average across time frames)
    noise_power = np.mean(np.abs(noise_stft) ** 2, axis=1, keepdims=True)

    # Spectral subtraction: reduce magnitude where noise dominates
    # We subtract noise power from signal power, floor at 0
    signal_power = np.abs(stft) ** 2
    clean_power = np.maximum(signal_power - noise_power, 0)

    # Reconstruct magnitude, keep original phase
    clean_magnitude = np.sqrt(clean_power)
    phase = np.angle(stft)
    clean_stft = clean_magnitude * np.exp(1j * phase)

    # Inverse STFT → back to time domain
    clean_samples = librosa.istft(clean_stft, hop_length=hop_length, length=len(samples))

    return clean_samples.astype(np.float32)
