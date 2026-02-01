import logging
import psutil
import os
from fastapi import APIRouter
from pydantic import BaseModel, Field
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
    cpu_percent: float = Field(
        ...,
        description="CPU usage percentage",
        title="CPU Usage (%)",
        ge=0.0,
        le=100.0
    )
    
    memory_percent: float = Field(
        ...,
        description="Memory usage percentage",
        title="Memory Usage (%)",
        ge=0.0,
        le=100.0
    )
    
    disk_percent: float = Field(
        ...,
        description="Disk usage percentage",
        title="Disk Usage (%)",
        ge=0.0,
        le=100.0
    )
    
    pid: int = Field(
        ...,
        description="Process ID of the running application",
        title="Process ID",
        ge=1
    )


@router.get("/info", summary="Get system information")
async def get_system_info() -> Dict[str, Any]:
    """
    Get comprehensive system and process resource information
    
    ## Returns:
    - **system**: Overall system resources
      - cpu_percent: System-wide CPU usage
      - memory: Total, available memory and usage percentage
      - disk: Total, free disk space and usage percentage
    - **process**: Current process resources
      - pid: Process ID
      - cpu_percent: CPU usage by this process
      - memory_mb: Memory used by this process in MB
      - threads: Number of threads running
    
    ## Use Cases:
    - Monitor system health
    - Check resource availability
    - Diagnose performance issues
    """
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


@router.get("/config", summary="Get application configuration")
async def get_config() -> Dict[str, Any]:
    """
    Get current application configuration settings
    
    ## Returns:
    - **api**: API server configuration
      - host: Server host address
      - port: Server port number
    - **llm**: Language model configuration
      - mode: LLM mode (local/openrouter)
      - model_path: Path to model file
    - **stt**: Speech-to-Text configuration
      - model_path: Path to STT model
    - **rvc**: RVC (Voice Conversion) configuration
      - base_url: RVC server URL
    - **memory**: Memory system configuration
      - cache_type: Type of cache (redis/in-memory)
      - redis: Redis connection details
    
    ## Use Cases:
    - Verify configuration
    - Debug connection issues
    - Check model paths
    """
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


@router.get("/health-detailed", summary="Comprehensive health check")
async def health_check_detailed():
    """
    Comprehensive health check for all system modules and resources
    
    ## Returns:
    - **status**: Overall system status (healthy/degraded/error)
    - **timestamp**: ISO timestamp of the health check
    - **modules**: Status of each system module
      - stt: Speech-to-Text module
      - dialog: Dialog/LLM module
      - tts: Text-to-Speech module
      - memory: Memory systems (short/long-term)
      - audio: Audio capture/playback
    - **resources**: System resource status
      - cpu: CPU usage and status
      - memory: RAM usage and status
      - disk: Disk usage and status
    - **models**: Model file availability
      - llm: Language model file status
      - stt: STT model file status
      - rvc: RVC configuration status
    
    ## Status Values:
    - **healthy**: All systems operational
    - **degraded**: Some non-critical issues detected
    - **error**: Critical failures present
    
    ## Resource Status:
    - **ok**: Usage below thresholds (CPU/Memory < 80%, Disk < 90%)
    - **warning**: Usage above warning thresholds
    
    ## Use Cases:
    - System monitoring dashboards
    - Pre-deployment checks
    - Troubleshooting module failures
    - Resource capacity planning
    """
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