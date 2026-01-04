import logging
import io
import threading
from pathlib import Path
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import soundfile as sf
import librosa
from modules.asr.transcriber import Transcriber
from setting import (
    STT_MODEL_PATH,
    VAD_THRESHOLD,
    SILENCE_CHUNKS_NEEDED,
    SAMPLE_RATE,
    MAX_UPLOAD_SIZE_MB,
    ALLOWED_AUDIO_FORMATS
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/stt", tags=["stt"])

_transcriber = None
_transcriber_lock = threading.Lock()

def get_transcriber() -> Transcriber:
    """Get or create the transcriber instance (thread-safe singleton)."""
    global _transcriber
    if _transcriber is None:
        with _transcriber_lock:
            if _transcriber is None:
                _transcriber = Transcriber(
                    vad_threshold=VAD_THRESHOLD,
                    stt_model_path=STT_MODEL_PATH,
                    sample_rate=SAMPLE_RATE,
                    silence_chunks_needed=SILENCE_CHUNKS_NEEDED
                )
                logger.info("Transcriber initialized successfully")
    return _transcriber

@router.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Transcribe an audio file to text.
    
    Args:
        file: Audio file (WAV, MP3, FLAC, OGG, M4A, AAC)
              Maximum size: 50MB
    
    Returns:
        JSON response with transcription result
    """
    try:
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in ALLOWED_AUDIO_FORMATS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format. Allowed formats: {', '.join(ALLOWED_AUDIO_FORMATS)}"
            )
        
        contents = await file.read()
        
        file_size_mb = len(contents) / (1024 * 1024)
        if file_size_mb > MAX_UPLOAD_SIZE_MB:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {MAX_UPLOAD_SIZE_MB}MB, received: {file_size_mb:.2f}MB"
            )
        
        audio_data, samplerate = sf.read(io.BytesIO(contents))
        
        if samplerate != SAMPLE_RATE:
            audio_data = librosa.resample(audio_data, orig_sr=samplerate, target_sr=SAMPLE_RATE)
        
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        transcriber = get_transcriber()
        result = transcriber.stt.transcribe(audio_data)
        
        logger.info(f"Successfully transcribed file: {file.filename}")
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "filename": file.filename,
                "transcription": result
            }
        )
    
    except Exception as e:
        logger.error(f"Error transcribing audio: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=400,
            detail=f"Error transcribing audio: {str(e)}"
        )