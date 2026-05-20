import argparse
import shutil
from pathlib import Path

import torch
from TTS.api import TTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a quick XTTS sample.")
    parser.add_argument(
        "--text",
        default="Hello. This is a test of the VoiceForge system.",
        help="Input text to synthesize.",
    )
    parser.add_argument(
        "--output",
        default="data/output.wav",
        help="Output wav path, relative to project root if not absolute.",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Language code (for example: en, es, fr). Defaults to en when available.",
    )
    parser.add_argument(
        "--speaker",
        default=None,
        help="Speaker name. Defaults to the first available speaker.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Inference device.",
    )
    parser.add_argument(
        "--model",
        default="tts_models/multilingual/multi-dataset/xtts_v2",
        help="TTS model name.",
    )
    return parser.parse_args()


def resolve_output_path(output: str) -> Path:
    output_path = Path(output)
    if output_path.is_absolute():
        return output_path
    project_root = Path(__file__).resolve().parent.parent
    return project_root / output_path


def get_available_speakers(tts: TTS) -> list[str]:
    if getattr(tts, "speakers", None):
        return list(tts.speakers)

    speaker_manager = getattr(getattr(tts.synthesizer, "tts_model", None), "speaker_manager", None)
    if speaker_manager is None:
        return []

    names = getattr(speaker_manager, "name_to_id", None)
    if names is None:
        return []

    return list(names)


def main() -> None:
    args = parse_args()

    print("Loading model...")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    tts = TTS(args.model).to(device)

    output_path = resolve_output_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Generating audio...")

    tts_kwargs = {
        "text": args.text,
        "file_path": str(output_path),
    }

    available_speakers = get_available_speakers(tts)
    speaker = args.speaker if args.speaker else (available_speakers[0] if available_speakers else None)
    if speaker:
        tts_kwargs["speaker"] = speaker
        print(f"Using speaker: {speaker}")

    available_languages = list(getattr(tts, "languages", []) or [])
    language = args.language if args.language else ("en" if "en" in available_languages else (available_languages[0] if available_languages else None))
    if language:
        tts_kwargs["language"] = language
        print(f"Using language: {language}")

    tts.tts_to_file(**tts_kwargs)

    root_output_path = Path(__file__).resolve().parent.parent / "output.wav"
    if output_path != root_output_path:
        shutil.copyfile(output_path, root_output_path)
        print(f"Compatibility copy written to: {root_output_path}")

    print(f"Done. File written to: {output_path}")


if __name__ == "__main__":
    main()