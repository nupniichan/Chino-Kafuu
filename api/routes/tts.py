import logging
import threading
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from src.modules.asr.tts import TTS

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/tts", tags=["TTS"])

_tts_instance = None
_tts_lock = threading.Lock()


def get_tts() -> TTS:
    global _tts_instance
    if _tts_instance is None:
        with _tts_lock:
            if _tts_instance is None:
                _tts_instance = TTS()
                logger.info("TTS initialized successfully")
    return _tts_instance


class TTSRequest(BaseModel):
    # Basic TTS settings
    text: str = Field(
        ..., 
        description="Text to synthesize into speech", 
        title="Text to Synthesize",
        min_length=1
    )
    
    tts_voice: str = Field(
        default="ja-JP-NanamiNeural", 
        description="Azure TTS voice name for synthesis", 
        title="TTS Voice",
        examples=["ja-JP-NanamiNeural", "en-US-JennyNeural", "vi-VN-HoaiMyNeural"]
    )
    
    # Voice model settings
    voice_model: str = Field(
        default="logs\\Chino-Kafuu\\Chino-Kafuu.pth", 
        description="Path to the RVC voice model file (.pth)", 
        title="Voice Model Path"
    )
    
    index_file: str = Field(
        default="logs\\Chino-Kafuu\\Chino-Kafuu.index", 
        description="Path to the voice model index file (.index)", 
        title="Index File Path"
    )
    
    # Input/Output settings
    param_1: str = Field(
        default="", 
        description="Input path for text file (leave empty for direct text input)", 
        title="Input Text File Path"
    )
    
    # TTS parameters - MUST BE INT
    param_4: int = Field(
        default=0, 
        description="Speed of TTS speech. 0 = normal speed, negative = slower, positive = faster", 
        title="TTS Speed",
        ge=-100,
        le=100
    )
    
    # Voice conversion parameters - MUST BE INT for pitch
    param_5: int = Field(
        default=2, 
        description="Pitch adjustment in semitones. Positive = higher pitch, negative = lower pitch", 
        title="Pitch (Semitones)",
        ge=-12,
        le=12
    )
    
    param_6: float = Field(
        default=0.5, 
        description="Search feature ratio for voice conversion. Higher = more like target voice, lower = preserve original characteristics", 
        title="Feature Search Ratio",
        ge=0.0,
        le=1.0
    )
    
    param_7: float = Field(
        default=1.0, 
        description="Volume envelope mixing scale. Controls how much of the original volume envelope to preserve", 
        title="Volume Envelope",
        ge=0.0,
        le=1.0
    )
    
    param_8: float = Field(
        default=0.5, 
        description="Protect voiceless consonants and breath sounds from being altered. Higher = more protection", 
        title="Protect Voiceless Consonants",
        ge=0.0,
        le=0.5
    )
    
    # Audio processing options
    param_14: bool = Field(
        default=False, 
        description="Split audio into smaller chunks for processing. Useful for long texts", 
        title="Split Audio"
    )
    
    # Autotune settings
    param_15: bool = Field(
        default=False, 
        description="Enable autotune for pitch correction and smoothing", 
        title="Enable Autotune"
    )
    
    param_16: float = Field(
        default=1.0, 
        description="Autotune strength/intensity (only works if Autotune is enabled). 0 = subtle, 1 = maximum", 
        title="Autotune Strength",
        ge=0.0,
        le=1.0
    )
    
    # Pitch extraction settings
    param_17: bool = Field(
        default=False, 
        description="Use proposed pitch extraction algorithm (alternative method)", 
        title="Use Proposed Pitch Extraction"
    )
    
    param_18: int = Field(
        default=255, 
        description="Threshold for proposed pitch extraction (only if proposed pitch is enabled)", 
        title="Proposed Pitch Threshold",
        ge=0,
        le=255
    )
    
    # Audio cleaning settings
    param_19: bool = Field(
        default=True, 
        description="Clean/denoise the audio output to remove background noise and artifacts", 
        title="Clean Audio"
    )
    
    param_20: float = Field(
        default=0.05, 
        description="Strength of audio cleaning/denoising (only if Clean Audio is enabled). Higher = more aggressive", 
        title="Clean Strength",
        ge=0.0,
        le=1.0
    )

    class Config:
        json_schema_extra = {
            "example": {
                "text": "こんにちは、私はチノです。今日はいい天気ですね。",
                "tts_voice": "ja-JP-NanamiNeural",
                "voice_model": "logs\\Chino-Kafuu\\Chino-Kafuu.pth",
                "index_file": "logs\\Chino-Kafuu\\Chino-Kafuu.index",
                "param_1": "",
                "param_4": 0,
                "param_5": 2,
                "param_6": 0.5,
                "param_7": 1.0,
                "param_8": 0.5,
                "param_14": False,
                "param_15": False,
                "param_16": 1.0,
                "param_17": False,
                "param_18": 255,
                "param_19": True,
                "param_20": 0.05
            }
        }


@router.post("/synthesize", summary="Synthesize Text to Speech")
async def synthesize_text(payload: TTSRequest):
    """
    Synthesize text to speech with voice conversion using RVC (Retrieval-based Voice Conversion)
    
    ## Main Parameters:
    - **text**: The text to synthesize (required)
    - **tts_voice**: Azure TTS voice to use as base
    - **voice_model**: RVC model for voice conversion
    
    ## Voice Adjustment:
    - **param_5 (Pitch)**: Adjust pitch in semitones (-12 to +12) - INTEGER only
    - **param_6 (Feature Ratio)**: Balance between original and target voice (0-1)
    - **param_8 (Protect Voiceless)**: Protect consonants from being altered (0-0.5)
    
    ## Audio Processing:
    - **param_19 (Clean Audio)**: Enable noise reduction
    - **param_15 (Autotune)**: Enable pitch correction
    - **param_14 (Split Audio)**: Process long audio in chunks
    
    ## Returns:
    - **success**: Boolean indicating success
    - **result**: Tuple of (info, audio_path)
    """
    try:
        if not payload.text:
            raise HTTPException(status_code=400, detail="Text is required")

        logger.info(f"Synthesizing text: {payload.text[:50]}...")
        tts = get_tts()

        # Ensure parameters are correct types
        result = tts.synthesize(
            text=payload.text,
            tts_voice=payload.tts_voice,
            voice_model=payload.voice_model,
            index_file=payload.index_file,
            param_1=payload.param_1,
            param_4=int(payload.param_4),  # Ensure int
            param_5=int(payload.param_5),  # Ensure int
            param_6=float(payload.param_6),  # Ensure float
            param_7=float(payload.param_7),  # Ensure float
            param_8=float(payload.param_8),  # Ensure float
            param_14=bool(payload.param_14),  # Ensure bool
            param_15=bool(payload.param_15),  # Ensure bool
            param_16=float(payload.param_16),  # Ensure float
            param_17=bool(payload.param_17),  # Ensure bool
            param_18=int(payload.param_18),  # Ensure int
            param_19=bool(payload.param_19),  # Ensure bool
            param_20=float(payload.param_20),  # Ensure float
        )

        logger.info(f"Synthesis completed: {result}")

        # Handle different return formats
        if isinstance(result, tuple) and len(result) == 2:
            info, audio_path = result
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "info": str(info),
                    "audio_path": str(audio_path),
                },
            )
        else:
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "result": str(result),
                },
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error synthesizing text: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error synthesizing text: {str(e)}",
        )


@router.get("/status", summary="Get TTS Service Status")
async def get_tts_status():
    """
    Check if the TTS service is ready and available
    
    ## Returns:
    - **status**: Service status (ready/error)
    - **server_url**: TTS server URL
    """
    try:
        _ = get_tts()
        return {
            "status": "ready",
            "server_url": "http://127.0.0.1:6969/",
            "model": "Chino-Kafuu",
        }
    except Exception as e:
        logger.error(f"TTS status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))