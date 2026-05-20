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
WHISPER_MODEL_SIZE = "base"         # tiny | base | small | medium
MIN_MATCH_SCORE = 0.6               # Below this = user read wrong prompt
WHISPER_DOWNLOAD_DIR = str(MODELS_DIR / "whisper")


# ── Model singleton ───────────────────────────────────────────────
# Same pattern as inference_service.py — load once, reuse.
_whisper_model: WhisperModel | None = None


def get_whisper_model() -> WhisperModel:
    """
    Load the Whisper model singleton.

    Uses CPU with int8 quantization — fast and low memory.
    int8 = weights stored as 8-bit integers instead of 32-bit floats.
    Roughly 4× smaller model, ~2× faster inference, tiny accuracy loss.
    """
    global _whisper_model

    if _whisper_model is None:
        device = "cuda" if runtime_config.cuda_available else "cpu"
        # int8 on CPU, float16 on GPU (both are faster than float32)
        compute_type = "float16" if runtime_config.cuda_available else "int8"

        logger.info(
            f"Loading Whisper {WHISPER_MODEL_SIZE} on {device} ({compute_type})"
        )
        _whisper_model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device=device,
            compute_type=compute_type,
            download_root=WHISPER_DOWNLOAD_DIR,
        )
        logger.info("Whisper model loaded")

    return _whisper_model


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
    model = get_whisper_model()

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
