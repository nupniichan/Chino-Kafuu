import logging
import psutil
import os
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict, Any

from setting import (
    LLM_MODE,
    LLM_MODEL_PATH,
    STT_MODEL_PATH,
    RVC_BASE_URL,
    MEMORY_CACHE,
    REDIS_HOST,
    REDIS_PORT,
    API_HOST,
    API_PORT
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/system", tags=["System"])


class SystemInfo(BaseModel):
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    pid: int


@router.get("/info")
async def get_system_info() -> Dict[str, Any]:
    try:
        process = psutil.Process(os.getpid())
        
        return {
            "system": {
                "cpu_percent": psutil.cpu_percent(interval=0.1),
                "memory": {
                    "total": psutil.virtual_memory().total,
                    "available": psutil.virtual_memory().available,
                    "percent": psutil.virtual_memory().percent
                },
                "disk": {
                    "total": psutil.disk_usage('/').total,
                    "free": psutil.disk_usage('/').free,
                    "percent": psutil.disk_usage('/').percent
                }
            },
            "process": {
                "pid": os.getpid(),
                "cpu_percent": process.cpu_percent(interval=0.1),
                "memory_mb": process.memory_info().rss / 1024 / 1024,
                "threads": process.num_threads()
            }
        }
    except Exception as e:
        logger.error(f"System info error: {e}")
        return {"error": str(e)}


@router.get("/config")
async def get_config() -> Dict[str, Any]:
    return {
        "api": {
            "host": API_HOST,
            "port": API_PORT
        },
        "llm": {
            "mode": LLM_MODE,
            "model_path": LLM_MODEL_PATH
        },
        "stt": {
            "model_path": STT_MODEL_PATH
        },
        "rvc": {
            "base_url": RVC_BASE_URL
        },
        "memory": {
            "cache_type": MEMORY_CACHE,
            "redis": {
                "host": REDIS_HOST,
                "port": REDIS_PORT
            }
        }
    }


@router.get("/health-detailed")
async def health_check_detailed():
    health_status = {
        "status": "healthy",
        "modules": {}
    }
    
    try:
        from modules.asr.transcriber import Transcriber
        health_status["modules"]["stt"] = "available"
    except Exception as e:
        health_status["modules"]["stt"] = f"error: {str(e)}"
        health_status["status"] = "degraded"
    
    try:
        from modules.dialog.orchestrator import DialogOrchestrator
        health_status["modules"]["dialog"] = "available"
    except Exception as e:
        health_status["modules"]["dialog"] = f"error: {str(e)}"
        health_status["status"] = "degraded"
    
    try:
        from modules.tts.tts_engine import TTSEngine
        health_status["modules"]["tts"] = "available"
    except Exception as e:
        health_status["modules"]["tts"] = f"error: {str(e)}"
        health_status["status"] = "degraded"
    
    try:
        from modules.memory.short_term import ShortTermMemory
        from modules.memory.long_term import LongTermMemory
        health_status["modules"]["memory"] = "available"
    except Exception as e:
        health_status["modules"]["memory"] = f"error: {str(e)}"
        health_status["status"] = "degraded"
    
    return health_status
