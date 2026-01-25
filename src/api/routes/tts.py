import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import asyncio

from modules.tts.tts_engine import TTSEngine
from setting import (
    RVC_BASE_URL,
    RVC_VOICE_MODEL,
    RVC_INDEX_FILE,
    RVC_OUTPUT_DIR
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/tts", tags=["TTS"])

_tts_engine = None


def get_tts_engine() -> TTSEngine:
    global _tts_engine
    if _tts_engine is None:
        _tts_engine = TTSEngine(
            rvc_base_url=RVC_BASE_URL,
            default_voice_model=RVC_VOICE_MODEL,
            default_index_file=RVC_INDEX_FILE,
            output_dir=RVC_OUTPUT_DIR
        )
        logger.info("TTS Engine initialized")
    return _tts_engine


class TTSRequest(BaseModel):
    text: str
    voice_model: Optional[str] = None
    pitch: Optional[int] = 0
    filter_radius: Optional[int] = 3
    index_rate: Optional[float] = 0.75
    volume_envelope: Optional[float] = 0.25
    protect: Optional[float] = 0.33


class TTSResponse(BaseModel):
    success: bool
    message: str
    audio_path: Optional[str] = None


@router.post("/synthesize", response_model=TTSResponse)
async def synthesize_speech(request: TTSRequest):
    try:
        engine = get_tts_engine()
        
        result = engine.synthesize(
            text=request.text,
            voice_model=request.voice_model,
            play_audio_after=False,
            pitch=request.pitch,
            filter_radius=request.filter_radius,
            index_rate=request.index_rate,
            volume_envelope=request.volume_envelope,
            protect=request.protect
        )
        
        return TTSResponse(
            success=True,
            message="Speech synthesized successfully",
            audio_path=result.export_audio_path
        )
        
    except Exception as e:
        logger.error(f"TTS synthesis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/synthesize-and-play", response_model=TTSResponse)
async def synthesize_and_play(request: TTSRequest):
    try:
        engine = get_tts_engine()
        
        result = engine.synthesize(
            text=request.text,
            voice_model=request.voice_model,
            play_audio_after=True,
            blocking_playback=False,
            pitch=request.pitch,
            filter_radius=request.filter_radius,
            index_rate=request.index_rate,
            volume_envelope=request.volume_envelope,
            protect=request.protect
        )
        
        return TTSResponse(
            success=True,
            message="Speech synthesized and playing",
            audio_path=result.export_audio_path
        )
        
    except Exception as e:
        logger.error(f"TTS synthesis and play error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_tts_status():
    try:
        engine = get_tts_engine()
        return {
            "status": "ready",
            "rvc_url": engine.rvc.base_url,
            "voice_model": engine.default_voice_model,
            "output_dir": str(engine.output_dir)
        }
    except Exception as e:
        logger.error(f"TTS status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
