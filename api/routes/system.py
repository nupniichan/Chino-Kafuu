import logging
import psutil
import os
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict, Any

from src.setting import (
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
    """Comprehensive health check for all system modules"""
    health_status = {
        "status": "healthy",
        "timestamp": None,
        "modules": {},
        "resources": {},
        "models": {}
    }
    
    # Import datetime for timestamp
    from datetime import datetime
    health_status["timestamp"] = datetime.now().isoformat()
    
    # Check STT module
    try:
        from src.modules.asr.transcriber import Transcriber
        from src.modules.asr.vad import VAD
        health_status["modules"]["stt"] = {
            "status": "available",
            "transcriber": "loaded",
            "vad": "loaded"
        }
    except Exception as e:
        health_status["modules"]["stt"] = {
            "status": "error",
            "error": str(e)
        }
        health_status["status"] = "degraded"
    
    # Check Dialog module
    try:
        from src.modules.dialog.orchestrator import DialogOrchestrator
        from src.modules.dialog.llm_wrapper import LLMWrapper
        health_status["modules"]["dialog"] = {
            "status": "available",
            "orchestrator": "loaded",
            "llm_wrapper": "loaded",
            "llm_type": LLM_MODE
        }
    except Exception as e:
        health_status["modules"]["dialog"] = {
            "status": "error",
            "error": str(e)
        }
        health_status["status"] = "degraded"
    
    # Check TTS module
    try:
        from src.modules.tts.tts_engine import TTSEngine
        from src.modules.tts.rvc_converter import RvcConverter
        health_status["modules"]["tts"] = {
            "status": "available",
            "tts_engine": "loaded",
            "rvc_converter": "loaded"
        }
    except Exception as e:
        health_status["modules"]["tts"] = {
            "status": "error",
            "error": str(e)
        }
        health_status["status"] = "degraded"
    
    # Check Memory modules
    try:
        from src.modules.memory.short_term import ShortTermMemory
        from src.modules.memory.long_term import LongTermMemory
        health_status["modules"]["memory"] = {
            "status": "available",
            "short_term": "loaded",
            "long_term": "loaded",
            "cache_type": MEMORY_CACHE
        }
    except Exception as e:
        health_status["modules"]["memory"] = {
            "status": "error",
            "error": str(e)
        }
        health_status["status"] = "degraded"
    
    # Check Audio modules
    try:
        from src.modules.audio.capture import AudioCapture
        from src.modules.audio.playback import AudioPlayer
        health_status["modules"]["audio"] = {
            "status": "available",
            "capture": "loaded",
            "playback": "loaded"
        }
    except Exception as e:
        health_status["modules"]["audio"] = {
            "status": "error",
            "error": str(e)
        }
        health_status["status"] = "degraded"
    
    # Check system resources
    try:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        health_status["resources"] = {
            "cpu": {
                "percent": cpu_percent,
                "status": "ok" if cpu_percent < 80 else "warning"
            },
            "memory": {
                "percent": memory.percent,
                "available_mb": memory.available / 1024 / 1024,
                "status": "ok" if memory.percent < 80 else "warning"
            },
            "disk": {
                "percent": disk.percent,
                "free_gb": disk.free / 1024 / 1024 / 1024,
                "status": "ok" if disk.percent < 90 else "warning"
            }
        }
        
        if cpu_percent >= 90 or memory.percent >= 90 or disk.percent >= 95:
            health_status["status"] = "degraded"
    except Exception as e:
        health_status["resources"] = {"error": str(e)}
    
    # Check model files existence
    try:
        import os
        health_status["models"] = {
            "llm": {
                "path": LLM_MODEL_PATH,
                "exists": os.path.exists(LLM_MODEL_PATH) if LLM_MODEL_PATH else False,
                "mode": LLM_MODE
            },
            "stt": {
                "path": STT_MODEL_PATH,
                "exists": os.path.exists(STT_MODEL_PATH) if STT_MODEL_PATH else False
            },
            "rvc": {
                "base_url": RVC_BASE_URL,
                "configured": bool(RVC_BASE_URL)
            }
        }
        
        # Check if critical models are missing
        if LLM_MODE == "local" and not health_status["models"]["llm"]["exists"]:
            health_status["status"] = "degraded"
            health_status["models"]["llm"]["warning"] = "Model file not found"
    except Exception as e:
        health_status["models"] = {"error": str(e)}
    
    return health_status
