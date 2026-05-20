from fastapi import APIRouter
from pydantic import BaseModel

from backend.core.settings import TTS_OUTPUT_DIR
from backend.services.inference_service import generate_speech

router = APIRouter()


class TTSRequest(BaseModel):
    text: str
    speaker_wav: str | None = None
    language: str = "en"


@router.post("/tts")
def tts(req: TTSRequest):
    output_path = str(TTS_OUTPUT_DIR / "output.wav")

    result_path = generate_speech(
        text=req.text,
        output_path=output_path,
        speaker_wav=req.speaker_wav,
        language=req.language,
    )

    return {
        "status": "success",
        "output": result_path,
    }

