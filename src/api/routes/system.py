"""
System API routes: health, config, EventBus stats, pipeline control.
"""
import logging
import os
from typing import Dict, Any
from datetime import datetime

import psutil
from fastapi import APIRouter, HTTPException

from src.core.bootstrap import registry, get_service
from src.setting import (
    LLM_MODE,
    LLM_MODEL_PATH,
    STT_MODEL_PATH,
    RVC_BASE_URL,
    MEMORY_CACHE,
    REDIS_HOST,
    REDIS_PORT,
    API_HOST,
    API_PORT,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/system", tags=["System"])


@router.get("/info", summary="Get system information")
async def get_system_info() -> Dict[str, Any]:
    try:
        process = psutil.Process(os.getpid())
        return {
            "system": {
                "cpu_percent": psutil.cpu_percent(interval=0.1),
                "memory": {
                    "total": psutil.virtual_memory().total,
                    "available": psutil.virtual_memory().available,
                    "percent": psutil.virtual_memory().percent,
                },
                "disk": {
                    "total": psutil.disk_usage("/").total,
                    "free": psutil.disk_usage("/").free,
                    "percent": psutil.disk_usage("/").percent,
                },
            },
            "process": {
                "pid": os.getpid(),
                "cpu_percent": process.cpu_percent(interval=0.1),
                "memory_mb": process.memory_info().rss / 1024 / 1024,
                "threads": process.num_threads(),
            },
        }
    except Exception as e:
        logger.error(f"System info error: {e}")
        return {"error": str(e)}


@router.get("/config", summary="Get application configuration")
async def get_config() -> Dict[str, Any]:
    return {
        "api": {"host": API_HOST, "port": API_PORT},
        "llm": {"mode": LLM_MODE, "model_path": LLM_MODEL_PATH},
        "stt": {"model_path": STT_MODEL_PATH},
        "rvc": {"base_url": RVC_BASE_URL},
        "memory": {
            "cache_type": MEMORY_CACHE,
            "redis": {"host": REDIS_HOST, "port": REDIS_PORT},
        },
    }


@router.get("/event-bus-stats", summary="Get EventBus statistics")
async def get_event_bus_stats():
    try:
        bus = get_service("event_bus")
        return {
            "stats": bus.get_stats(),
            "recent_events": bus.get_history(limit=20),
        }
    except KeyError:
        return {"status": "not_initialized"}


@router.get("/health-detailed", summary="Comprehensive health check")
async def health_check_detailed():
    health = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {},
        "resources": {},
    }

    try:
        bus = get_service("event_bus")
        health["services"]["event_bus"] = {
            "status": "active",
            **bus.get_stats(),
        }
    except KeyError:
        health["services"]["event_bus"] = {"status": "not_initialized"}
        health["status"] = "degraded"

    try:
        engine = get_service("dialog_engine")
        health["services"]["dialog_engine"] = {
            "status": "active",
            "is_processing": engine.is_processing,
        }
    except KeyError:
        health["services"]["dialog_engine"] = {"status": "not_initialized"}
        health["status"] = "degraded"

    try:
        mm = get_service("memory_manager")
        health["services"]["memory"] = {
            "status": "active",
            **mm.get_stats(),
        }
    except KeyError:
        health["services"]["memory"] = {"status": "not_initialized"}
        health["status"] = "degraded"

    try:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()
        health["resources"] = {
            "cpu": {"percent": cpu_percent, "status": "ok" if cpu_percent < 80 else "warning"},
            "memory": {
                "percent": mem.percent,
                "available_mb": mem.available / 1024 / 1024,
                "status": "ok" if mem.percent < 80 else "warning",
            },
        }
    except Exception as e:
        health["resources"] = {"error": str(e)}

    health["models"] = {
        "llm": {
            "path": LLM_MODEL_PATH,
            "exists": os.path.exists(LLM_MODEL_PATH) if LLM_MODEL_PATH else False,
            "mode": LLM_MODE,
        },
        "stt": {
            "path": STT_MODEL_PATH,
            "exists": os.path.exists(STT_MODEL_PATH) if STT_MODEL_PATH else False,
        },
    }

    return health


@router.post("/pipeline/start", summary="Start real-time audio pipeline")
async def start_pipeline():
    """Start the Mic -> VAD -> STT -> EventBus real-time loop."""
    try:
        from src.core.pipeline import RealtimePipeline

        bus = get_service("event_bus")

        try:
            pipeline = get_service("pipeline")
        except KeyError:
            pipeline = RealtimePipeline(event_bus=bus)
            pipeline.register()
            registry._services["pipeline"] = pipeline

        if pipeline.is_running:
            return {"status": "already_running"}

        await pipeline.start()
        return {"status": "started"}
    except Exception as e:
        logger.error(f"Pipeline start error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/pipeline/stop", summary="Stop real-time audio pipeline")
async def stop_pipeline():
    try:
        pipeline = get_service("pipeline")
        pipeline.stop()
        return {"status": "stopped"}
    except KeyError:
        return {"status": "not_running"}
    except Exception as e:
        logger.error(f"Pipeline stop error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
