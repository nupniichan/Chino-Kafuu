import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import asyncio

from src.modules.tts.tts_engine import TTSEngine
from src.setting import (
    RVC_BASE_URL,
    RVC_VOICE_MODEL,
    RVC_INDEX_FILE,
    RVC_OUTPUT_DIR,
    TTS_VOICE,
    TTS_SPEED,
    RVC_PITCH,
    RVC_SEARCH_FEATURE_RATIO,
    RVC_VOLUME_ENVELOPE,
    RVC_PROTECT_VOICELESS,
    RVC_SPLIT_AUDIO,
    RVC_AUTOTUNE,
    RVC_AUTOTUNE_STRENGTH,
    RVC_PROPOSED_PITCH,
    RVC_PROPOSED_PITCH_THRESHOLD,
    RVC_CLEAN_AUDIO,
    RVC_CLEAN_STRENGTH
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
    tts_voice: Optional[str] = None
    tts_speed: Optional[float] = None
    input_text_file_path: Optional[str] = ""
    pitch: Optional[float] = None
    search_feature_ratio: Optional[float] = None
    volume_envelope: Optional[float] = None
    protect_voiceless_consonants: Optional[float] = None
    split_audio: Optional[bool] = None
    autotune: Optional[bool] = None
    autotune_strength: Optional[float] = None
    proposed_pitch: Optional[bool] = None
    proposed_pitch_threshold: Optional[float] = None
    clean_audio: Optional[bool] = None
    clean_strength: Optional[float] = None
    # Optional overrides
    index_file: Optional[str] = None


class TTSResponse(BaseModel):
    success: bool
    message: str
    audio_path: Optional[str] = None


@router.post("/synthesize", response_model=TTSResponse)
async def synthesize_speech(request: TTSRequest):
    try:
        engine = get_tts_engine()
        
        # Build kwargs with defaults from settings
        kwargs = {
            "tts_voice": request.tts_voice or TTS_VOICE,
            "tts_speed": request.tts_speed if request.tts_speed is not None else TTS_SPEED,
            "input_text_file_path": request.input_text_file_path or "",
            "pitch": request.pitch if request.pitch is not None else RVC_PITCH,
            "search_feature_ratio": request.search_feature_ratio if request.search_feature_ratio is not None else RVC_SEARCH_FEATURE_RATIO,
            "volume_envelope": request.volume_envelope if request.volume_envelope is not None else RVC_VOLUME_ENVELOPE,
            "protect_voiceless_consonants": request.protect_voiceless_consonants if request.protect_voiceless_consonants is not None else RVC_PROTECT_VOICELESS,
            "split_audio": request.split_audio if request.split_audio is not None else RVC_SPLIT_AUDIO,
            "autotune": request.autotune if request.autotune is not None else RVC_AUTOTUNE,
            "autotune_strength": request.autotune_strength if request.autotune_strength is not None else RVC_AUTOTUNE_STRENGTH,
            "proposed_pitch": request.proposed_pitch if request.proposed_pitch is not None else RVC_PROPOSED_PITCH,
            "proposed_pitch_threshold": request.proposed_pitch_threshold if request.proposed_pitch_threshold is not None else RVC_PROPOSED_PITCH_THRESHOLD,
            "clean_audio": request.clean_audio if request.clean_audio is not None else RVC_CLEAN_AUDIO,
            "clean_strength": request.clean_strength if request.clean_strength is not None else RVC_CLEAN_STRENGTH,
        }
        
        if request.index_file:
            kwargs["index_file"] = request.index_file
        
        result = engine.synthesize(
            text=request.text,
            play_audio_after=False,
            **kwargs
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
        
        kwargs = {
            "tts_voice": request.tts_voice or TTS_VOICE,
            "tts_speed": request.tts_speed if request.tts_speed is not None else TTS_SPEED,
            "input_text_file_path": request.input_text_file_path or "",
            "pitch": request.pitch if request.pitch is not None else RVC_PITCH,
            "search_feature_ratio": request.search_feature_ratio if request.search_feature_ratio is not None else RVC_SEARCH_FEATURE_RATIO,
            "volume_envelope": request.volume_envelope if request.volume_envelope is not None else RVC_VOLUME_ENVELOPE,
            "protect_voiceless_consonants": request.protect_voiceless_consonants if request.protect_voiceless_consonants is not None else RVC_PROTECT_VOICELESS,
            "split_audio": request.split_audio if request.split_audio is not None else RVC_SPLIT_AUDIO,
            "autotune": request.autotune if request.autotune is not None else RVC_AUTOTUNE,
            "autotune_strength": request.autotune_strength if request.autotune_strength is not None else RVC_AUTOTUNE_STRENGTH,
            "proposed_pitch": request.proposed_pitch if request.proposed_pitch is not None else RVC_PROPOSED_PITCH,
            "proposed_pitch_threshold": request.proposed_pitch_threshold if request.proposed_pitch_threshold is not None else RVC_PROPOSED_PITCH_THRESHOLD,
            "clean_audio": request.clean_audio if request.clean_audio is not None else RVC_CLEAN_AUDIO,
            "clean_strength": request.clean_strength if request.clean_strength is not None else RVC_CLEAN_STRENGTH,
        }
        
        if request.index_file:
            kwargs["index_file"] = request.index_file
        
        result = engine.synthesize(
            text=request.text,
            play_audio_after=True,
            blocking_playback=False,
            **kwargs
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
