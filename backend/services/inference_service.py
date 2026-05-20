from pathlib import Path

from TTS.api import TTS

from backend.core.logger import logger
from backend.core.settings import TTS_MODEL_NAME
from backend.core.settings import runtime_config

tts_model = None


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
    """Load the TTS model singleton, respecting low-VRAM config."""
    global tts_model

    if tts_model is None:
        device = "cuda" if runtime_config.cuda_available else "cpu"

        if runtime_config.low_vram_mode:
            logger.info(f"Low VRAM mode enabled — loading model in fp16 on {device}")
            tts_model = TTS(model_name=TTS_MODEL_NAME, gpu=runtime_config.cuda_available)
            # Force half precision to reduce VRAM usage
            if hasattr(tts_model.synthesizer, "tts_model"):
                tts_model.synthesizer.tts_model.half()
        else:
            logger.info(f"Loading TTS model on device: {device}")
            tts_model = TTS(model_name=TTS_MODEL_NAME).to(device)

        logger.info("TTS model loaded successfully")

    return tts_model


def generate_speech(
    text: str,
    output_path: str,
    speaker_wav: str | None = None,
    speaker: str | None = None,
    language: str = "en"
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

    Returns:
        Absolute path to the generated .wav file.

    Raises:
        ValueError: If synthesis fails for any reason.
    """
    try:
        model = load_model()

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        if speaker_wav:
            # Primary path — voice cloning
            ref = Path(speaker_wav)
            if not ref.exists():
                raise ValueError(f"Reference audio not found: {speaker_wav}")

            logger.info(f"Voice cloning from reference: {ref.name}")
            model.tts_to_file(
                text=text,
                speaker_wav=str(ref),
                language=language,
                file_path=str(output_file),
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

            model.tts_to_file(
                text=text,
                speaker=selected,
                language=selected_language,
                file_path=str(output_file),
            )

        logger.info(f"Speech generated successfully: {output_file}")
        return str(output_file.resolve())

    except ValueError:
        raise
    except Exception as e:
        logger.error(f"Speech generation failed: {e}")
        raise ValueError(f"Failed to generate speech: {e}") from e