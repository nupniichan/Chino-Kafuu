import logging
from fastapi import APIRouter
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/base", tags=["Base"])

@router.get("/health")
async def health_check():
    """
    Check if the system is running and all modules are available.
    Verifies that STT and VAD modules can be imported successfully.
    """
    try:
        from modules.asr.transcriber import Transcriber
        from modules.asr.stt import STT
        from modules.asr.vad import VAD
        
        logger.info("Health check: All modules available")
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "healthy",
                "message": "System is running normally",
                "modules": {
                    "stt": "available",
                    "vad": "available",
                    "transcriber": "available"
                }
            }
        )
    except ImportError as e:
        logger.error(f"Health check failed: {str(e)}")
        
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "message": f"Module import failed: {str(e)}"
            }
        )