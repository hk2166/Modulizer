from pathlib import Path
from dataclasses import dataclass

import numpy as np
import soundfile as sf
from TTS.api import TTS

from backend.core.logger import logger
from backend.core.settings import TTS_MODEL_NAME
from backend.core.settings import runtime_config

tts_model = None


# ── Synthesis tuning parameters ───────────────────────────────────────────
# These control pace and delivery. XTTS does NOT copy speaking rate from the
# reference clip — rhythm is generated each call and governed by these knobs:
#
#   speed              0.5–2.0  global time-stretch. <1 slows, >1 speeds up.
#   temperature        0.1–1.0  delivery variation. Low = flat/robotic/even,
#                               high = natural rhythm but glitch risk.
#   length_penalty     >0       higher = terser/clipped, lower = drawn out.
#   repetition_penalty 1.0–15.0 stops repeated syllables/vowels. XTTS needs
#                               a high value (default 5.0 here).
#
# Defaults are tuned to feel natural without glitching. The "too rushed /
# too robotic" fix is usually speed≈0.92 + temperature≈0.8.
@dataclass
class SynthesisParams:
    speed: float = 1.0
    temperature: float = 0.75
    length_penalty: float = 1.0
    repetition_penalty: float = 5.0
    top_k: int = 50
    top_p: float = 0.85

    def as_kwargs(self) -> dict:
        """The subset XTTS's tts_to_file / inference accept as kwargs."""
        return {
            "speed": self.speed,
            "temperature": self.temperature,
            "length_penalty": self.length_penalty,
            "repetition_penalty": self.repetition_penalty,
            "top_k": self.top_k,
            "top_p": self.top_p,
        }


# ── XTTS token-limit helpers ──────────────────────────────────────────────
# XTTS's GPT core has a hard 400-token limit per call. We pre-split long
# text into safe chunks, synthesise each, and concatenate the wavs so the
# user can generate arbitrarily long speech without hitting the limit.
#
# Why pre-split rather than rely on tts_to_file's built-in split_sentences?
# tts_to_file passes split_sentences=True but the 400-token check fires
# *before* the split, on the full text. Pre-splitting at our layer is the
# only way to avoid the RuntimeError on long inputs.
#
# split_sentence's `text_split_length` is in characters, not tokens.
# ~7 chars/token for English → 380 tokens ≈ 2,660 chars.
# We use 1,400 chars — conservative enough to cover CJK/Devanagari which
# pack more meaning per char and therefore more tokens per char.
_SPLIT_CHAR_LIMIT = 1_400   # chars per chunk fed to split_sentence
_TOKEN_LIMIT = 380           # hard ceiling checked after sentence split


def _get_vocab_path() -> str | None:
    """Find the XTTS vocab.json already downloaded by the TTS singleton."""
    try:
        from TTS.utils.manage import ModelManager
        manager = ModelManager()
        model_path, _, _ = manager.download_model(
            "tts_models/multilingual/multi-dataset/xtts_v2"
        )
        vocab = Path(model_path) / "vocab.json"
        return str(vocab) if vocab.exists() else None
    except Exception:
        return None


def _split_text(text: str, language: str) -> list[str]:
    """
    Split `text` into chunks each under _TOKEN_LIMIT XTTS tokens.

    Strategy:
      1. Use XTTS's own sentence-aware split_sentence (spacy sentencizer)
         with _SPLIT_CHAR_LIMIT chars per group. This handles typical prose.
      2. For any chunk still over the token limit (long run-on sentences),
         fall back to splitting at clause markers (;, ,) then words.

    Returns a list of non-empty strings, each safe to pass to tts_to_file.
    """
    try:
        from TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer, split_sentence

        vocab = _get_vocab_path()
        tokenizer = VoiceBpeTokenizer(vocab_file=vocab) if vocab else None

        def _token_count(s: str) -> int:
            if tokenizer is None:
                # Rough estimate: 1 token ≈ 4 chars
                return len(s) // 4
            try:
                return len(tokenizer.encode(s, language))
            except Exception:
                return len(s) // 4

        # Step 1: sentence-level split
        try:
            sentences = split_sentence(text, language, _SPLIT_CHAR_LIMIT)
        except Exception:
            import re
            sentences = re.split(r"(?<=[.?!।])\s+", text)

        sentences = [s.strip() for s in sentences if s.strip()]

        # Step 2: sub-chunk any sentence that's still over the token limit
        final_chunks: list[str] = []
        for sentence in sentences:
            if _token_count(sentence) <= _TOKEN_LIMIT:
                final_chunks.append(sentence)
                continue

            # Try clause markers in order of preference
            sub_chunks: list[str] = [sentence]
            for sep in (";", "،", ",", " "):
                new_sub: list[str] = []
                for chunk in sub_chunks:
                    if _token_count(chunk) <= _TOKEN_LIMIT:
                        new_sub.append(chunk)
                        continue
                    parts = chunk.split(sep)
                    current = ""
                    for part in parts:
                        candidate = (current + sep + part).strip() if current else part.strip()
                        if _token_count(candidate) <= _TOKEN_LIMIT:
                            current = candidate
                        else:
                            if current:
                                new_sub.append(current)
                            current = part.strip()
                    if current:
                        new_sub.append(current)
                sub_chunks = new_sub
                if all(_token_count(c) <= _TOKEN_LIMIT for c in sub_chunks):
                    break

            final_chunks.extend(c for c in sub_chunks if c)

        return final_chunks or [text]

    except Exception as e:
        logger.warning(f"_split_text: split failed ({e}), returning original text")
        return [text]


def _synth_chunks(
    model,
    chunks: list[str],
    output_file: Path,
    *,
    speaker_wav: str | None = None,
    speaker: str | None = None,
    language: str = "en",
    params: SynthesisParams | None = None,
) -> None:
    """
    Synthesise one or more text chunks and write the result to output_file.
    Single chunks are written directly; multiple chunks are concatenated.
    `params` carries speed / temperature / penalties; defaults if None.
    """
    tuning = (params or SynthesisParams()).as_kwargs()

    if len(chunks) == 1:
        model.tts_to_file(
            text=chunks[0],
            speaker_wav=speaker_wav,
            speaker=speaker,
            language=language,
            file_path=str(output_file),
            split_sentences=False,   # we already split
            **tuning,
        )
        return

    import tempfile
    segments = []
    sr = None
    with tempfile.TemporaryDirectory() as tmp:
        for i, chunk in enumerate(chunks):
            tmp_path = Path(tmp) / f"chunk_{i:03d}.wav"
            model.tts_to_file(
                text=chunk,
                speaker_wav=speaker_wav,
                speaker=speaker,
                language=language,
                file_path=str(tmp_path),
                split_sentences=False,
                **tuning,
            )
            audio, file_sr = sf.read(str(tmp_path))
            segments.append(audio)
            if sr is None:
                sr = file_sr

    combined = np.concatenate(segments)
    sf.write(str(output_file), combined, sr)


def get_available_speakers(tts: TTS) -> list[str]:
    """Extract available speakers from the TTS model."""
    if getattr(tts, "speakers", None):
        return list(tts.speakers)

    speaker_manager = getattr(
        getattr(tts.synthesizer, "tts_model", None),
        "speaker_manager",
        None
    )
    if speaker_manager is None:
        return []

    names = getattr(speaker_manager, "name_to_id", None)
    if names is None:
        return []

    return list(names)


def get_available_languages(tts: TTS) -> list[str]:
    """Extract available languages from the TTS model."""
    return list(getattr(tts, "languages", []) or [])


def load_model() -> TTS:
    """
    Load the TTS model singleton.

    A note on low-VRAM mode: we used to call `.half()` here on the
    assumption that fp16 weights would help on small GPUs. It doesn't —
    XTTS's audio pipeline (speaker encoder, dvae, HiFi-GAN decoder) feeds
    fp32 tensors into the model, and a fp16-weight / fp32-input mismatch
    raises at the first matmul. Inference fits in ~2 GB of VRAM on stock
    fp32, well inside a GTX 1650's 4 GB. We let the training pipeline
    handle mixed precision (see training_config.py) where it actually
    matters.
    """
    global tts_model

    if tts_model is None:
        device = "cuda" if runtime_config.cuda_available else "cpu"
        logger.info(f"Loading TTS model on device: {device}")
        tts_model = TTS(model_name=TTS_MODEL_NAME).to(device)
        logger.info("TTS model loaded successfully")

    return tts_model


def generate_speech(
    text: str,
    output_path: str,
    speaker_wav: str | None = None,
    speaker: str | None = None,
    language: str = "en",
    params: SynthesisParams | None = None,
) -> str:
    """
    Generate speech from text using XTTS v2.

    Two modes:
      - Voice cloning:   pass speaker_wav= (path to reference .wav) — primary product path
      - Built-in voice:  pass speaker= (named voice from base model) — testing/fallback only

    Args:
        text:         Input text to synthesize.
        output_path:  Path to save the output .wav file.
        speaker_wav:  Path to reference audio for voice cloning (preferred).
        speaker:      Built-in speaker name (fallback when no reference audio).
        language:     BCP-47 language code, defaults to 'en'.
        params:       Synthesis tuning (speed, temperature, penalties).
                      Defaults to SynthesisParams() if None.

    Returns:
        Absolute path to the generated .wav file.

    Raises:
        ValueError: If synthesis fails for any reason.
    """
    params = params or SynthesisParams()
    try:
        model = load_model()

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # App-layer text normalization before XTTS tokenization.
        # Handles what the tokenizer doesn't: Indic number expansion,
        # Latin-word transliteration for Hindi/Indic text, NFC normalization.
        # Returns None for English and unsupported languages (no-op).
        from backend.audio.text_cleaners import get_cleaners
        cleaner = get_cleaners(language)
        if cleaner:
            original = text
            text = cleaner(text)
            if text != original:
                logger.info(
                    f"Text normalized for lang={language}: "
                    f"{original!r} → {text!r}"
                )

        if speaker_wav:
            # Primary path — voice cloning
            ref = Path(speaker_wav)
            if not ref.exists():
                raise ValueError(f"Reference audio not found: {speaker_wav}")

            chunks = _split_text(text, language)
            logger.info(
                f"Voice cloning from reference: {ref.name} "
                f"({len(chunks)} chunk{'s' if len(chunks) != 1 else ''})"
            )
            _synth_chunks(
                model, chunks, output_file,
                speaker_wav=str(ref),
                language=language,
                params=params,
            )

        else:
            # Fallback path — built-in speaker (useful for M0 smoke tests)
            available_speakers = get_available_speakers(model)
            selected = speaker or (available_speakers[0] if available_speakers else None)

            if selected:
                logger.info(f"Using built-in speaker: {selected}")
            else:
                logger.warning("No speaker_wav and no built-in speakers found — attempting bare synthesis")

            available_languages = get_available_languages(model)
            selected_language = language if language in available_languages else (
                available_languages[0] if available_languages else language
            )

            chunks = _split_text(text, selected_language)
            logger.info(f"Built-in speaker synthesis: {len(chunks)} chunk(s)")
            _synth_chunks(
                model, chunks, output_file,
                speaker=selected,
                language=selected_language,
                params=params,
            )

        logger.info(f"Speech generated successfully: {output_file}")
        return str(output_file.resolve())

    except ValueError:
        raise
    except Exception as e:
        logger.error(f"Speech generation failed: {e}")
        raise ValueError(f"Failed to generate speech: {e}") from e


def generate_speech_from_checkpoint(
    text: str,
    output_path: str,
    checkpoint_path: str,
    config_path: str,
    speaker_wav: str,
    language: str = "en",
    params: SynthesisParams | None = None,
) -> str:
    """
    Generate speech using a fine-tuned Voice Profile checkpoint.

    Loads XTTS from the fine-tuned weights rather than the base model.
    Does NOT reuse the `tts_model` singleton — fine-tuned weights are
    a different model variant and must be loaded separately.

    Args:
        text:            Input text to synthesise.
        output_path:     Path to save the output .wav.
        checkpoint_path: Path to best_model.pth (or any checkpoint .pth).
        config_path:     Path to config.json in the same run directory.
        speaker_wav:     Reference clip for speaker conditioning.
        language:        BCP-47 language code.

    Returns:
        Absolute path to the generated .wav file.

    Raises:
        ValueError: on any synthesis failure.
    """
    try:
        from pathlib import Path as _Path
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts
        import torch

        from backend.audio.text_cleaners import get_cleaners

        ref = _Path(speaker_wav)
        if not ref.exists():
            raise ValueError(f"Reference audio not found: {speaker_wav}")

        out = _Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        # Load config from the training run directory.
        config = XttsConfig()
        config.load_json(config_path)

        # Load the model from the fine-tuned checkpoint.
        device = "cuda" if runtime_config.cuda_available else "cpu"
        model = Xtts.init_from_config(config)
        model.load_checkpoint(config, checkpoint_path=checkpoint_path, eval=True)
        model.to(device)

        logger.info(
            f"Voice Profile synthesis: checkpoint={_Path(checkpoint_path).name}, "
            f"ref={ref.name}"
        )

        # Apply text normalization same as the base-model path.
        cleaner = get_cleaners(language)
        if cleaner:
            text = cleaner(text)

        # Compute conditioning latents from the reference clip.
        gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
            audio_path=str(ref)
        )

        # Inference.
        params = params or SynthesisParams()
        out_dict = model.inference(
            text=text,
            language=language,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            temperature=params.temperature,
            length_penalty=params.length_penalty,
            repetition_penalty=params.repetition_penalty,
            top_k=params.top_k,
            top_p=params.top_p,
            speed=params.speed,
        )

        import soundfile as sf
        import numpy as np
        wav = out_dict.get("wav") or out_dict.get("audio")
        if wav is None:
            raise ValueError("Model returned no audio data.")
        wav_np = np.array(wav, dtype=np.float32)
        sf.write(str(out), wav_np, config.audio.output_sample_rate)

        logger.info(f"Voice Profile speech generated: {out}")
        return str(out.resolve())

    except ValueError:
        raise
    except Exception as e:
        logger.error(f"Voice Profile speech generation failed: {e}")
        raise ValueError(f"Failed to generate speech from checkpoint: {e}") from e
