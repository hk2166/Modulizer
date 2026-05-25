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

WHY 22.05 kHz (and not 24 kHz):
  XTTS v2's GPT/dvae stack tokenises audio at 22050 Hz — that's the rate
  every component upstream of HiFi-GAN expects. The decoder synthesises
  at 24 kHz, but that's set independently in `XttsAudioConfig.output_sample_rate`.
  If we wrote 24 kHz wavs and told the dataset loader they were 22050,
  the model would receive an ~8% pitch-shifted signal and produce
  metallic, robotic output. Match the GPT internal rate exactly here.

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
  Off by default. When enabled, runs DeepFilterNet 3 — a small neural
  denoiser — on CPU. Handles non-stationary noise (clicks, rustles,
  distant voices) and preserves formants better than spectral subtraction.
  Adds ~200 ms per clip on a modern CPU. The model is lazy-loaded on
  first use so the import cost only hits projects that actually denoise.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa

from backend.core.logger import logger
from backend.core.settings import DATA_DIR


# DeepFilterNet model singleton. CPU-only by deliberate choice:
#   • Keeps the GPU 100% available for XTTS on low-VRAM cards.
#   • Preprocessing is offline (one shot per clip), so latency doesn't matter.
#   • Same code path works on machines without a GPU at all.
# Lazy-initialised on first call to _get_dfn() so the torch + df imports
# don't run unless someone actually denoises.
_dfn_state = None  # tuple[model, df_state, sr] once loaded


# ── Configuration ─────────────────────────────────────────────────
# 22050 Hz is the rate the XTTS GPT/dvae trains at internally. The
# decoder (HiFi-GAN) synthesises at 24 kHz and that's set independently
# in `XttsAudioConfig.output_sample_rate`. If we resample the dataset to
# 24 kHz here, every training step pays a torchaudio resample back down
# to 22050 — and worse, an off-by-~8% mismatch between the rate written
# to disk and the rate the dataset loader assumes will produce metallic
# pitch-shifted audio. Match the GPT internal rate exactly.
TARGET_SAMPLE_RATE = 22050          # XTTS GPT / DVAE native rate
TARGET_PEAK_DBFS = -3.0             # Normalize peaks to -3 dBFS
TRIM_TOP_DB = 30                    # Silence threshold for trimming (dB below peak)
                                    # Higher = more aggressive trimming


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
        denoise:     Run a CPU-side neural denoiser (DeepFilterNet 3) before
                normalization. Off by default; turn on for clips with
                persistent background noise (HVAC, fan, distant traffic).
                Adds ~200 ms per clip on a modern CPU.

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

    # ── Step 2: Resample to 22.05 kHz ─────────────────────────────
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

    # ── Step 4: Optional neural denoise (DeepFilterNet 3, CPU) ────
    if denoise:
        samples = _dfn_denoise(samples, TARGET_SAMPLE_RATE)
        logger.info("Applied neural denoising (DFN3)")

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


def _get_dfn():
    """
    Lazy-load DFN3 once per process on CPU.

    Imports torch and the `df` package on first call. Subsequent calls
    return the cached (model, df_state, sr) tuple. Module top-level
    imports stay light so projects that never denoise don't pay the
    torch + DFN import cost (~3 s on cold start).
    """
    global _dfn_state
    if _dfn_state is not None:
        return _dfn_state

    import torch
    from df import init_df

    logger.info("Loading DeepFilterNet (CPU)")
    model, df_state, _ = init_df()
    model.eval()
    model.to(torch.device("cpu"))
    sr = df_state.sr()
    _dfn_state = (model, df_state, sr)
    logger.info(f"DeepFilterNet ready (model SR = {sr} Hz)")
    return _dfn_state


def _dfn_denoise(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Neural denoising via DeepFilterNet 3.

    Better than the old spectral subtraction at the things that matter
    for voice cloning: handles non-stationary noise (clicks, rustles,
    distant voices) and preserves formants. Replaced the old hand-rolled
    `_spectral_denoise` which stripped formants and made clones sound
    hollow.

    Runs on CPU only (~50 MB RAM, ~200 ms per 6 s clip on a modern CPU)
    so it doesn't compete with XTTS for VRAM on low-VRAM cards.

    Args:
        samples:     float32 mono audio in [-1, 1].
        sample_rate: rate of `samples`. DFN expects 48 kHz internally;
                     we resample if needed and resample back at the end
                     so the rest of the pipeline doesn't need to change.

    Returns:
        Denoised float32 audio at the **input** sample rate.
    """
    import torch
    from df.enhance import enhance

    if len(samples) == 0:
        return samples

    model, df_state, dfn_sr = _get_dfn()

    # DFN expects 48 kHz. Up-resample if needed; we'll bring it back at the end.
    if sample_rate != dfn_sr:
        upsampled = librosa.resample(samples, orig_sr=sample_rate, target_sr=dfn_sr)
    else:
        upsampled = samples

    # DFN's enhance() takes a torch tensor with shape (channels, samples).
    audio_t = torch.from_numpy(upsampled).unsqueeze(0)
    with torch.no_grad():
        enhanced_t = enhance(model, df_state, audio_t)
    enhanced = enhanced_t.squeeze(0).cpu().numpy().astype(np.float32)

    # Resample back so downstream code (peak normalize, write) sees the same SR.
    if sample_rate != dfn_sr:
        enhanced = librosa.resample(enhanced, orig_sr=dfn_sr, target_sr=sample_rate)

    return enhanced