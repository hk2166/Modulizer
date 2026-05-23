"""
dataset_builder.py — turn processed project clips into an XTTS-ready dataset.

WHAT THIS DOES
──────────────
Coqui TTS (the library that hosts XTTS) doesn't accept a directory of .wav
files directly for fine-tuning. It needs a *dataset* in a specific layout
with a metadata file that pairs each audio clip with its transcript.

We produce the **LJSpeech format** because:
  - It's the simplest of the supported formats (one CSV, two columns).
  - It's the format used in every Coqui XTTS fine-tuning example.
  - It's pipe-delimited, so it's tolerant of commas in transcripts.

OUTPUT LAYOUT
─────────────
    data/projects/{project_id}/dataset/
    ├── wavs/
    │   ├── 0001.wav        ← copy of each processed clip, renamed
    │   ├── 0002.wav
    │   └── ...
    ├── metadata.csv        ← LJSpeech-format manifest
    └── manifest.json       ← our own metadata (clip count, lang, etc.)

metadata.csv format (one line per clip):
    {wav_basename}|{normalized_text}|{normalized_text}

  Yes, the text is duplicated. That's an LJSpeech quirk: column 2 is
  the original text, column 3 is the "normalized" text (numbers spelled
  out, abbreviations expanded). For our purposes they're the same.

WHY WE COPY THE CLIPS INSTEAD OF SYMLINKING
────────────────────────────────────────────
1. Symlinks break inside zip exports (an M3 task).
2. The dataset becomes self-contained and portable — you can hand the
   folder to any other machine and it'll train.
3. Disk cost is negligible (a few MB per minute of audio).

PIPELINE
────────
  for each clip in processed/:
      1. Get / generate transcript (call transcriber.py if missing)
      2. Filter out low-confidence transcripts (training noise)
      3. Sanitize text (strip newlines, normalize whitespace, escape pipes)
      4. Copy clip to dataset/wavs/{NNNN}.wav
      5. Append metadata line

  → write metadata.csv
  → write manifest.json
  → optional: train/eval split (default 95/5)

KEY CONCEPTS
────────────
• Transcripts: each .wav must have its spoken text. We use Whisper to
  auto-transcribe if the user didn't provide one. For Voice Profile
  training (M2), this is the entire reason we have the transcriber.

• Train/eval split: ML training needs unseen examples to measure quality.
  We hold back 5% (~1 of every 20 clips) as eval data. Standard practice.

• Text sanitization: pipe `|` is the column separator, so any literal
  pipe in the transcript would corrupt the file. We replace it with a
  space. Newlines get the same treatment (one clip = one line).
"""

from __future__ import annotations

import csv
import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from backend.audio.transcriber import transcribe_clip
from backend.core.logger import logger
from backend.core.settings import DATA_DIR


# ── Configuration ─────────────────────────────────────────────────
DEFAULT_EVAL_FRACTION = 0.05    # 5% of clips held back for evaluation
MIN_TRANSCRIPT_WORDS = 2        # Below this, the transcript is probably wrong
MIN_AUDIO_SECONDS = 1.0         # Reject anything shorter — training waste
MAX_AUDIO_SECONDS = 30.0        # XTTS reference cap; longer = wasted samples
SPEAKER_NAME_DEFAULT = "voiceforge"   # Single-speaker datasets use a fixed name

# Golden-reference scoring: the duration sweet spot for an XTTS reference
# clip. Long enough to capture prosody, short enough to fit the conditioning
# window without wasted compute. 6 s is the empirical centre.
GOLDEN_DURATION_IDEAL_LO = 5.0
GOLDEN_DURATION_IDEAL_HI = 7.0
GOLDEN_DURATION_CENTER = 6.0
GOLDEN_DURATION_WEIGHT = 0.7
GOLDEN_STABILITY_WEIGHT = 0.3


# ══════════════════════════════════════════════════════════════════
# Result types
# ══════════════════════════════════════════════════════════════════


@dataclass
class ClipEntry:
    """One clip's contribution to the dataset."""
    clip_id: str
    original_path: str            # Source path in processed/
    dataset_path: str             # Destination path in dataset/wavs/
    text: str                     # Transcript (sanitized)
    duration_s: float
    auto_transcribed: bool        # True if Whisper produced this text


@dataclass
class DatasetBuildResult:
    """Outcome of building a dataset for a project."""
    success: bool
    dataset_dir: str | None = None
    metadata_csv: str | None = None
    manifest_json: str | None = None
    train_count: int = 0
    eval_count: int = 0
    skipped_count: int = 0
    skipped_reasons: dict[str, int] = field(default_factory=dict)
    total_duration_s: float = 0.0
    language: str = "en"
    error: str | None = None


# ══════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════


def build_dataset(
    project_id: str,
    language: str = "en",
    transcripts: dict[str, str] | None = None,
    eval_fraction: float = DEFAULT_EVAL_FRACTION,
    speaker_name: str = SPEAKER_NAME_DEFAULT,
) -> DatasetBuildResult:
    """
    Build an XTTS-ready dataset from the project's processed clips.

    Args:
        project_id:    Project to build for.
        language:      Language code for transcription (e.g. 'en', 'hi').
        transcripts:   Optional pre-existing {clip_id: text} map. Any clip
                       not in this dict gets auto-transcribed.
        eval_fraction: Portion held back as the eval set. 0 disables split.
        speaker_name:  LJSpeech is single-speaker but Coqui's variant lets
                       you tag clips. Stored in the manifest for later.

    Returns:
        DatasetBuildResult with paths and counts. On failure, success=False
        and `error` carries a friendly message.
    """
    transcripts = transcripts or {}

    project_dir = DATA_DIR / "projects" / project_id
    processed_dir = project_dir / "processed"
    if not processed_dir.exists():
        return DatasetBuildResult(
            success=False,
            error="No processed clips found. Record or import audio first.",
        )

    clips = sorted(processed_dir.glob("*.wav"))
    if not clips:
        return DatasetBuildResult(
            success=False,
            error="No processed clips to build a dataset from.",
        )

    logger.info(f"dataset_builder: building for project={project_id}, {len(clips)} clips")

    # ── Set up output directory ───────────────────────────────────
    dataset_dir = project_dir / "dataset"
    wavs_dir = dataset_dir / "wavs"

    # Wipe any previous dataset to avoid stale clips. Cheap and safe —
    # the source data lives in processed/, this is just a derivative.
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    wavs_dir.mkdir(parents=True, exist_ok=True)

    # ── Iterate clips, build entries ──────────────────────────────
    entries: list[ClipEntry] = []
    skipped: dict[str, int] = {}

    for idx, clip_path in enumerate(clips, start=1):
        clip_id = clip_path.stem  # filename without extension
        logger.info(f"dataset_builder: processing clip {idx}/{len(clips)}: {clip_id}")

        # 1. Duration sanity
        duration = _audio_duration_s(clip_path)
        if duration < MIN_AUDIO_SECONDS:
            skipped["too_short"] = skipped.get("too_short", 0) + 1
            continue
        if duration > MAX_AUDIO_SECONDS:
            skipped["too_long"] = skipped.get("too_long", 0) + 1
            continue

        # 2. Get transcript
        text, auto = _get_transcript(clip_id, clip_path, language, transcripts)

        if text is None:
            skipped["no_transcript"] = skipped.get("no_transcript", 0) + 1
            continue

        sanitized = _sanitize_text(text)
        if len(sanitized.split()) < MIN_TRANSCRIPT_WORDS:
            skipped["short_transcript"] = skipped.get("short_transcript", 0) + 1
            continue

        # 3. Stage the clip into wavs/{NNNN}.wav (1-indexed, 4 digits)
        dataset_wav = wavs_dir / f"{idx:04d}.wav"
        shutil.copyfile(clip_path, dataset_wav)

        entries.append(ClipEntry(
            clip_id=clip_id,
            original_path=str(clip_path.resolve()),
            dataset_path=str(dataset_wav.resolve()),
            text=sanitized,
            duration_s=round(duration, 2),
            auto_transcribed=auto,
        ))

    if not entries:
        # Don't leave an empty dataset_dir lying around; clean up.
        shutil.rmtree(dataset_dir, ignore_errors=True)
        return DatasetBuildResult(
            success=False,
            error="No clips were usable for training (after duration / transcript checks).",
            skipped_count=sum(skipped.values()),
            skipped_reasons=skipped,
        )

    # ── Train / eval split ────────────────────────────────────────
    train_entries, eval_entries = _split_train_eval(entries, eval_fraction)

    # ── Write metadata.csv (LJSpeech format) ──────────────────────
    metadata_csv = dataset_dir / "metadata.csv"
    _write_ljspeech_metadata(metadata_csv, train_entries)

    # If we have an eval set, write it to its own file. Coqui supports a
    # `meta_file_val` argument, so eval lives in metadata_eval.csv.
    eval_csv = None
    if eval_entries:
        eval_csv = dataset_dir / "metadata_eval.csv"
        _write_ljspeech_metadata(eval_csv, eval_entries)

    # ── Manifest (our own bookkeeping) ────────────────────────────
    manifest_path = dataset_dir / "manifest.json"
    total_duration = sum(e.duration_s for e in entries)
    _write_manifest(
        manifest_path,
        project_id=project_id,
        speaker_name=speaker_name,
        language=language,
        train_entries=train_entries,
        eval_entries=eval_entries,
        total_duration_s=total_duration,
        skipped_reasons=skipped,
    )

    logger.info(
        f"dataset_builder: built dataset for project={project_id} "
        f"(train={len(train_entries)}, eval={len(eval_entries)}, "
        f"skipped={sum(skipped.values())}, total={total_duration:.1f}s)"
    )

    return DatasetBuildResult(
        success=True,
        dataset_dir=str(dataset_dir.resolve()),
        metadata_csv=str(metadata_csv.resolve()),
        manifest_json=str(manifest_path.resolve()),
        train_count=len(train_entries),
        eval_count=len(eval_entries),
        skipped_count=sum(skipped.values()),
        skipped_reasons=skipped,
        total_duration_s=round(total_duration, 2),
        language=language,
    )


# ══════════════════════════════════════════════════════════════════
# Internals
# ══════════════════════════════════════════════════════════════════


def _audio_duration_s(path: Path) -> float:
    """Read just the WAV header to get duration, no full decode."""
    import soundfile as sf
    info = sf.info(str(path))
    return info.frames / info.samplerate


def _get_transcript(
    clip_id: str,
    clip_path: Path,
    language: str,
    transcripts: dict[str, str],
) -> tuple[str | None, bool]:
    """
    Resolve the transcript for a clip.

    Order of preference:
      1. User-provided text in the `transcripts` dict (clip_id → text)
      2. Whisper auto-transcription (faster-whisper)

    Returns:
        (text, auto_transcribed) — or (None, False) if nothing usable.
    """
    if clip_id in transcripts and transcripts[clip_id].strip():
        return transcripts[clip_id], False

    # Auto-transcribe. We pass an empty `expected_text` because we don't
    # care about a match score here — we just want what was said.
    try:
        result = transcribe_clip(clip_path, expected_text="", language=language)
    except Exception as e:
        logger.warning(f"dataset_builder: transcription failed for {clip_id}: {e}")
        return None, False

    text = result.transcribed_text.strip()
    if not text:
        return None, False
    return text, True


# Pipe is the LJSpeech column separator; literal pipes in text would
# corrupt the file. Newlines would split a single record into two.
_BAD_CHARS_RE = re.compile(r"[|\r\n\t]+")
_WHITESPACE_RE = re.compile(r"\s+")


def _sanitize_text(text: str) -> str:
    """
    Make text safe for LJSpeech metadata.

    - Replace pipe / newline / tab with a space (avoids file corruption).
    - Collapse runs of whitespace to single spaces.
    - Strip leading/trailing whitespace.
    """
    text = _BAD_CHARS_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def _split_train_eval(
    entries: list[ClipEntry],
    eval_fraction: float,
) -> tuple[list[ClipEntry], list[ClipEntry]]:
    """
    Split entries into train / eval lists.

    We use a deterministic alternating-sample approach instead of random:
      - Reproducible (same input → same split)
      - No need for a seeded RNG
      - Even distribution: eval samples are spread across the dataset
        instead of clustered at the end.

    Example with 20 clips and 5% eval:
      eval_fraction=0.05 → keep_every = 20 (every 20th clip)
      Eval set: clips at index 0, 20, 40, ... (only one for n=20)
      Train set: everything else
    """
    if eval_fraction <= 0 or len(entries) < 4:
        # Too few clips to meaningfully split — give everything to train.
        return list(entries), []

    # Choose a stride: e.g. 0.05 → 1 in 20 clips goes to eval.
    keep_every = max(2, int(round(1.0 / eval_fraction)))

    train, evals = [], []
    for i, entry in enumerate(entries):
        # Use middle-of-stride as eval index to avoid taking the very first
        # clip (which is often the user's "test" recording).
        if i % keep_every == keep_every // 2:
            evals.append(entry)
        else:
            train.append(entry)

    # Edge case: tiny dataset where stride > len(entries) → no evals chosen
    if not evals and len(entries) >= 4:
        # Pull the middle clip as a single eval sample
        mid = len(entries) // 2
        evals = [train.pop(mid)] if train else []

    return train, evals


def _write_ljspeech_metadata(path: Path, entries: list[ClipEntry]) -> None:
    """
    Write LJSpeech-format metadata.csv.

    Format: `<wav_basename>|<text>|<normalized_text>\n`
    No header. UTF-8. Pipe separator. No quoting.

    The "normalized" text in column 3 is supposed to be the speakable
    expansion ("Dr." → "Doctor", "5" → "five"). For our auto-generated
    transcripts these are already plain text, so we just duplicate.
    """
    # We deliberately avoid `csv.writer` because LJSpeech doesn't quote
    # fields, and csv.writer would helpfully add quotes around any text
    # containing commas. Manual write keeps the format strict.
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for e in entries:
            wav_basename = Path(e.dataset_path).stem  # e.g. "0001"
            f.write(f"{wav_basename}|{e.text}|{e.text}\n")


def _write_manifest(
    path: Path,
    *,
    project_id: str,
    speaker_name: str,
    language: str,
    train_entries: list[ClipEntry],
    eval_entries: list[ClipEntry],
    total_duration_s: float,
    skipped_reasons: dict[str, int],
) -> None:
    """
    Write our own manifest.json — not consumed by Coqui, but useful for
    the UI ("here's what's in this dataset") and for export/import later.
    """
    manifest = {
        "project_id": project_id,
        "format": "ljspeech",
        "speaker_name": speaker_name,
        "language": language,
        "total_clips": len(train_entries) + len(eval_entries),
        "train_count": len(train_entries),
        "eval_count": len(eval_entries),
        "total_duration_s": round(total_duration_s, 2),
        "skipped_reasons": skipped_reasons,
        "clips": [
            {
                "clip_id": e.clip_id,
                "wav": Path(e.dataset_path).name,
                "text": e.text,
                "duration_s": e.duration_s,
                "auto_transcribed": e.auto_transcribed,
                "split": "train",
            }
            for e in train_entries
        ] + [
            {
                "clip_id": e.clip_id,
                "wav": Path(e.dataset_path).name,
                "text": e.text,
                "duration_s": e.duration_s,
                "auto_transcribed": e.auto_transcribed,
                "split": "eval",
            }
            for e in eval_entries
        ],
    }
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


# ══════════════════════════════════════════════════════════════════
# Golden reference selection
# ══════════════════════════════════════════════════════════════════


def select_golden_reference(processed_dir: str | Path) -> str | None:
    """
    Pick the best clip to use as a default voice-cloning reference.

    Used as a safety net by the inference service: when the user calls
    `generate_speech` without an explicit `speaker_wav`, we want to grab
    the clip from this project that's most likely to produce a clean,
    consistent clone — not just whichever clip ffmpeg dropped first.

    Scoring (weighted):
      • Duration fit (0.7 weight)
        XTTS clones best from references in the 5–7 s range — long enough
        to capture prosody, short enough to stay inside the conditioning
        window. Anything in [5,7] gets the max score; everything else is
        penalised by its distance from 6 s.

      • Pitch stability (0.3 weight)
        Computed as 1 / variance of YIN F0 across voiced frames. Lower
        F0 variance = steadier pitch = cleaner reference. Unvoiced frames
        return NaN from YIN, so we filter them before computing variance —
        otherwise a clip that's 90% silence would look perfectly stable.

    We deliberately ignore amplitude: the preprocessor already normalises
    every clip, so loudness no longer carries useful information.

    Args:
        processed_dir: absolute or relative path to a directory of .wav
                       files (typically `data/projects/{id}/processed/`).

    Returns:
        Absolute path to the highest-scoring clip, or None if the
        directory has no readable WAVs.
    """
    import numpy as np
    import librosa
    import soundfile as sf

    processed = Path(processed_dir)
    if not processed.exists():
        logger.warning(f"select_golden_reference: {processed} does not exist")
        return None

    wavs = sorted(processed.glob("*.wav"))
    if not wavs:
        logger.warning(f"select_golden_reference: no .wav files in {processed}")
        return None

    best_path: Path | None = None
    best_score = float("-inf")

    for wav_path in wavs:
        try:
            audio, native_sr = sf.read(str(wav_path), dtype="float32")
        except Exception as e:
            logger.warning(f"select_golden_reference: skipping {wav_path.name}: {e}")
            continue

        # Multi-channel safety net — pick the first channel.
        if audio.ndim > 1:
            audio = audio[:, 0]

        duration = len(audio) / native_sr if native_sr else 0.0
        if duration <= 0:
            continue

        # ── Duration fit ────────────────────────────────────────
        if GOLDEN_DURATION_IDEAL_LO <= duration <= GOLDEN_DURATION_IDEAL_HI:
            duration_score = 10.0
        else:
            # Linear penalty per second from the centre. Floor at 0 so
            # very long or very short clips don't drag the total negative.
            duration_score = max(0.0, 10.0 - abs(GOLDEN_DURATION_CENTER - duration))

        # ── Pitch stability ─────────────────────────────────────
        # YIN is unreliable on near-silent frames and returns NaN there.
        # We filter to voiced frames before computing variance so a
        # mostly-silent clip can't masquerade as "perfectly stable".
        try:
            f0 = librosa.yin(
                audio,
                fmin=librosa.note_to_hz("C2"),
                fmax=librosa.note_to_hz("C6"),
                sr=native_sr,
            )
            voiced = f0[np.isfinite(f0) & (f0 > 0)]
            if voiced.size < 8:
                # Not enough voiced frames to draw a conclusion either way.
                stability_score = 0.0
            else:
                variance = float(np.var(voiced))
                stability_score = 1.0 / (variance + 1e-5)
        except Exception as e:
            logger.warning(
                f"select_golden_reference: pitch analysis failed for "
                f"{wav_path.name}: {e}"
            )
            stability_score = 0.0

        total = (duration_score * GOLDEN_DURATION_WEIGHT) + \
                (stability_score * GOLDEN_STABILITY_WEIGHT)

        logger.info(
            f"select_golden_reference: {wav_path.name} "
            f"dur={duration:.2f}s dur_score={duration_score:.2f} "
            f"stab_score={stability_score:.4f} total={total:.4f}"
        )

        if total > best_score:
            best_score = total
            best_path = wav_path

    if best_path is None:
        # Every clip failed to read or score — fall back to the first one
        # so callers always get *something* usable rather than a None they
        # have to handle. The cost of a bad reference is a worse clone;
        # the cost of None is a 500 from the inference endpoint.
        best_path = wavs[0]
        logger.warning(
            f"select_golden_reference: scoring failed for all clips, "
            f"falling back to {best_path.name}"
        )

    logger.info(f"select_golden_reference: chose {best_path.name}")
    return str(best_path.resolve())
