"""
transcriber.py — Whisper-based clip QA: did the user read the right prompt?

HOW IT WORKS:
─────────────
1. Load a Whisper model (tiny or base — fast, good enough for QA).
2. Transcribe the processed audio clip.
3. Compare the transcription to the expected script text.
4. Return a similarity score and pass/fail.

WHY FASTER-WHISPER:
  faster-whisper is a reimplementation of OpenAI Whisper using CTranslate2,
  a C++ inference engine. It's 4× faster than the original and uses less RAM.
  Same accuracy, just optimized for CPU/GPU inference.

SIMILARITY ALGORITHM — Word Error Rate (WER):
  WER is the standard metric for speech recognition quality.
  WER = (substitutions + deletions + insertions) / total_reference_words

  We use 1 - WER as our "match score":
    1.0 = perfect match
    0.8 = 80% of words correct (acceptable)
    0.5 = half the words wrong (fail)

  We use a simpler approximation: word-level Jaccard similarity.
  Jaccard = |intersection| / |union| of word sets.
  This ignores word order but is fast and good enough for QA.

  Example:
    Expected: "the quick brown fox"
    Got:      "the quick brown dog"
    Intersection: {the, quick, brown} = 3 words
    Union: {the, quick, brown, fox, dog} = 5 words
    Score: 3/5 = 0.6

MODEL SIZES (tradeoff: speed vs accuracy):
  tiny   — 39M params, ~32× realtime on CPU, good enough for QA
  base   — 74M params, ~16× realtime on CPU, better accuracy
  small  — 244M params, ~6× realtime on CPU, near-perfect
  medium — 769M params, ~2× realtime on CPU, very accurate

  For clip QA we use "base" — fast enough, accurate enough.
"""

import re
import string
from dataclasses import dataclass
from pathlib import Path

from faster_whisper import WhisperModel

from backend.core.logger import logger
from backend.core.settings import MODELS_DIR, runtime_config


# ── Configuration ─────────────────────────────────────────────────
# Whisper model size by language. Hindi (and most non-English languages)
# need at least 'medium' — 'base' has very high WER on Devanagari text,
# producing garbled transcripts that poison the dataset csv when used for
# fine-tuning. English clip QA is fine on 'base'.
#
# Sizes ship as separate weights; we lazy-load each on first use so the
# user only pays the disk cost (medium ~1.5 GB) when they actually pick
# a non-English language.
WHISPER_SIZE_BY_LANG = {
    "en": "base",
    "hi": "medium",
}
WHISPER_DEFAULT_SIZE = "base"      # for languages we haven't tuned yet
MIN_MATCH_SCORE = 0.6              # Below this = user read wrong prompt
WHISPER_DOWNLOAD_DIR = str(MODELS_DIR / "whisper")


# ── VRAM headroom needed per Whisper size (GB) ────────────────────
# Approximate total VRAM a CTranslate2 float16 model needs to load AND
# run a beam-search transcribe without OOM-ing. Deliberately conservative
# (includes workspace + beam-5 activations). If the GPU has less than this,
# we transcribe on CPU instead — transcription is a one-time QA/transcript
# pass and must never compete with XTTS training for scarce VRAM.
_WHISPER_VRAM_NEED_GB = {
    "tiny": 1.0,
    "base": 1.5,
    "small": 3.0,
    "medium": 5.0,
    "large": 10.0,
    "large-v2": 10.0,
    "large-v3": 10.0,
}


# ── Model singletons (one per size) ───────────────────────────────
# Keyed by Whisper size, not language — both 'en' and any other 'base'
# language share a single loaded model. Saves memory in the common case.
_whisper_models: dict[str, WhisperModel] = {}


def _whisper_size_for_language(language: str) -> str:
    """Pick the right Whisper model size for the given language."""
    return WHISPER_SIZE_BY_LANG.get(language, WHISPER_DEFAULT_SIZE)


def _pick_whisper_device(size: str) -> tuple[str, str]:
    """
    Choose (device, compute_type) for a given Whisper size.

    Use the GPU only when it has enough VRAM to hold the model with room to
    run; otherwise fall back to CPU int8. A 4 GB card, for instance, can't
    fit 'medium' in float16 — trying anyway OOMs on every clip.
    """
    if runtime_config.cuda_available:
        need = _WHISPER_VRAM_NEED_GB.get(size, 5.0)
        if runtime_config.vram_gb >= need:
            return "cuda", "float16"
        logger.info(
            f"Whisper {size} needs ~{need:.0f} GB VRAM but only "
            f"{runtime_config.vram_gb:.1f} GB available — transcribing on CPU "
            f"to keep the GPU free for training."
        )
    return "cpu", "int8"


def _load_whisper(size: str, device: str, compute_type: str) -> WhisperModel | None:
    """Try to construct a WhisperModel. Returns None on failure (e.g. OOM)."""
    logger.info(f"Loading Whisper {size} on {device} ({compute_type})")
    try:
        model = WhisperModel(
            size,
            device=device,
            compute_type=compute_type,
            download_root=WHISPER_DOWNLOAD_DIR,
        )
        logger.info(f"Whisper {size} loaded on {device}")
        return model
    except Exception as e:
        logger.warning(f"Whisper {size} load on {device} failed: {e}")
        return None


def get_whisper_model(language: str = "en") -> WhisperModel:
    """
    Load (or reuse) the Whisper model appropriate for `language`.

    Different languages map to different model sizes via WHISPER_SIZE_BY_LANG.
    The first call for a given size pays the load cost; subsequent calls
    return the cached instance.

    Device is chosen by VRAM fit (`_pick_whisper_device`). If a GPU load
    fails anyway (OOM, driver hiccup), we fall back to CPU rather than letting
    the failure cascade into a broken, empty dataset.
    """
    size = _whisper_size_for_language(language)

    if size in _whisper_models:
        return _whisper_models[size]

    device, compute_type = _pick_whisper_device(size)
    model = _load_whisper(size, device, compute_type)

    # Belt-and-suspenders: if the GPU load failed, retry on CPU. Covers the
    # case where VRAM looked sufficient but allocation failed at construction.
    if model is None and device == "cuda":
        logger.warning(f"Whisper {size} didn't fit on the GPU — retrying on CPU.")
        model = _load_whisper(size, "cpu", "int8")

    if model is None:
        raise RuntimeError(
            f"Couldn't load the Whisper '{size}' model on any device."
        )

    _whisper_models[size] = model
    return model


@dataclass
class TranscriptionResult:
    """Result of transcribing and QA-checking a clip."""
    transcribed_text: str           # What Whisper heard
    expected_text: str              # What the user was supposed to say
    match_score: float              # 0.0–1.0, higher = better match
    passed: bool                    # True if match_score >= MIN_MATCH_SCORE
    language: str                   # Detected language code (e.g. "en")
    message: str                    # User-friendly feedback


def transcribe_clip(
    audio_path: str | Path,
    expected_text: str,
    language: str = "en",
) -> TranscriptionResult:
    """
    Transcribe a clip and check it against the expected script text.

    Args:
        audio_path:    Path to the processed .wav file.
        expected_text: The prompt text the user was supposed to read.
        language:      Expected language code (helps Whisper accuracy).

    Returns:
        TranscriptionResult with transcription, score, and feedback.
    """
    audio_path = Path(audio_path)
    model = get_whisper_model(language)

    # ── Transcribe ────────────────────────────────────────────────
    # model.transcribe() returns (segments_iterator, info)
    # segments is a lazy iterator — we consume it to get the text.
    # vad_filter=True uses Voice Activity Detection to skip silence,
    # which reduces hallucinations (Whisper sometimes invents text
    # for silent audio).
    try:
        segments, info = model.transcribe(
            str(audio_path),
            language=language,
            vad_filter=True,
            beam_size=5,            # Higher = more accurate, slower
        )
        # Join all segment texts into one string
        transcribed_text = " ".join(seg.text.strip() for seg in segments)
    except Exception as e:
        logger.error(f"Transcription failed for {audio_path.name}: {e}")
        return TranscriptionResult(
            transcribed_text="",
            expected_text=expected_text,
            match_score=0.0,
            passed=False,
            language=language,
            message="We had trouble understanding that recording. Please try again.",
        )

    logger.info(f"Transcribed: '{transcribed_text[:80]}...'")

    # ── Compare to expected text ──────────────────────────────────
    match_score = _word_similarity(transcribed_text, expected_text)
    passed = match_score >= MIN_MATCH_SCORE

    message = _build_feedback_message(passed, match_score, transcribed_text)

    logger.info(
        f"QA result: clip={audio_path.name}, "
        f"score={match_score:.2f}, passed={passed}"
    )

    return TranscriptionResult(
        transcribed_text=transcribed_text,
        expected_text=expected_text,
        match_score=round(match_score, 3),
        passed=passed,
        language=info.language,
        message=message,
    )


# ══════════════════════════════════════════════════════════════════
# Internal helpers
# ══════════════════════════════════════════════════════════════════


def _normalize_text(text: str) -> set[str]:
    """
    Normalize text to a set of lowercase words, no punctuation.

    "Hello, World!" → {"hello", "world"}

    WHY A SET: We're doing Jaccard similarity (word overlap), not
    sequence matching. Sets give us fast intersection/union.
    The tradeoff: we lose word order and repetition info, but for
    QA purposes ("did they say the right words?") that's fine.
    """
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Lowercase and split on whitespace
    words = text.lower().split()
    # Filter out empty strings and very short words (articles, etc.)
    return set(w for w in words if len(w) > 1)


def _word_similarity(text_a: str, text_b: str) -> float:
    """
    Compute Jaccard similarity between two texts at the word level.

    Jaccard(A, B) = |A ∩ B| / |A ∪ B|

    Returns 0.0 (no overlap) to 1.0 (identical word sets).
    """
    words_a = _normalize_text(text_a)
    words_b = _normalize_text(text_b)

    if not words_a and not words_b:
        return 1.0  # Both empty — technically identical
    if not words_a or not words_b:
        return 0.0  # One is empty

    intersection = words_a & words_b     # Words in both
    union = words_a | words_b            # All unique words

    return len(intersection) / len(union)


def _build_feedback_message(passed: bool, score: float, transcribed: str) -> str:
    """Build a user-friendly message based on the QA result."""
    if passed:
        if score >= 0.9:
            return "That sounded great — we caught every word clearly."
        return "Good take — we understood most of what you said."

    if score < 0.2:
        return (
            "We couldn't make out what you said. "
            "Try speaking more clearly and closer to the microphone."
        )
    if not transcribed.strip():
        return "We didn't hear anything. Make sure your microphone is working."

    return (
        "It sounds like you may have read a different line. "
        "Please read the text shown on screen."
    )


# ══════════════════════════════════════════════════════════════════
# Prompt loader — reads from data/scripts/default_prompts.json
# ══════════════════════════════════════════════════════════════════

import json
from backend.core.settings import DEFAULT_PROMPTS_FILE


def load_prompts(language: str = "en") -> list[dict]:
    """
    Load prompts for a given language from the default prompts file.

    Args:
        language: Language code — "en" or "hi".

    Returns:
        List of prompt dicts with "id", "text", and "focus" keys.

    Example:
        prompts = load_prompts("hi")
        print(prompts[0]["text"])  # कच्चा पापड़, पक्का पापड़...
    """
    with open(DEFAULT_PROMPTS_FILE, encoding="utf-8") as f:
        data = json.load(f)

    languages = data.get("languages", {})
    if language not in languages:
        available = list(languages.keys())
        raise ValueError(
            f"Language '{language}' not found in prompts file. "
            f"Available: {available}"
        )

    return languages[language]["prompts"]


def get_prompt_by_id(prompt_id: int, language: str = "en") -> dict | None:
    """
    Get a single prompt by its ID.

    Returns None if not found.
    """
    prompts = load_prompts(language)
    for prompt in prompts:
        if prompt["id"] == prompt_id:
            return prompt
    return None
