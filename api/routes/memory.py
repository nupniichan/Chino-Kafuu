import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from src.modules.memory.short_term import ShortTermMemory
from src.modules.memory.long_term import LongTermMemory
from src.setting import (
    SHORT_TERM_MEMORY_SIZE,
    MEMORY_CACHE,
    REDIS_HOST,
    REDIS_PORT,
    REDIS_DB
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/memory", tags=["Memory"])

_short_memory = None
_long_memory = None


def get_short_memory() -> ShortTermMemory:
    global _short_memory
    if _short_memory is None:
        _short_memory = ShortTermMemory(
            max_size=SHORT_TERM_MEMORY_SIZE,
            storage_type=MEMORY_CACHE,
            redis_host=REDIS_HOST,
            redis_port=REDIS_PORT,
            redis_db=REDIS_DB
        )
        logger.info("Short-term memory initialized")
    return _short_memory


def get_long_memory() -> LongTermMemory:
    global _long_memory
    if _long_memory is None:
        _long_memory = LongTermMemory()
        logger.info("Long-term memory initialized")
    return _long_memory


class MessageEntry(BaseModel):
    message: str
    emotion: str = "normal"
    lang: str = "en"
    source: str = "api"


class ResponseEntry(BaseModel):
    text: str
    emotion: str = "normal"
    lang: str = "en"


@router.post("/short-term/add-user")
async def add_user_message(entry: MessageEntry):
    try:
        memory = get_short_memory()
        result = memory.add_user_message(
            message=entry.message,
            emotion=entry.emotion,
            lang=entry.lang,
            source=entry.source
        )
        return {"success": True, "entry": result}
    except Exception as e:
        logger.error(f"Add user message error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/short-term/add-chino")
async def add_chino_response(entry: ResponseEntry):
    try:
        memory = get_short_memory()
        result = memory.add_chino_response(
            text=entry.text,
            emotion=entry.emotion,
            lang=entry.lang
        )
        return {"success": True, "entry": result}
    except Exception as e:
        logger.error(f"Add chino response error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/short-term/buffer")
async def get_buffer():
    try:
        memory = get_short_memory()
        buffer = memory.get_recent_messages()
        return {
            "success": True,
            "session_id": memory.current_session_id,
            "buffer_size": len(buffer),
            "buffer": buffer
        }
    except Exception as e:
        logger.error(f"Get buffer error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/short-term/clear")
async def clear_buffer():
    try:
        memory = get_short_memory()
        memory.clear()
        return {"success": True, "message": "Buffer cleared"}
    except Exception as e:
        logger.error(f"Clear buffer error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/short-term/new-session")
async def start_new_session():
    try:
        memory = get_short_memory()
        session_id = memory.start_new_session()
        return {"success": True, "session_id": session_id}
    except Exception as e:
        logger.error(f"Start new session error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/long-term/summaries")
async def get_summaries(
    session_id: Optional[str] = None,
    limit: int = 10,
    min_importance: float = 0.0
):
    try:
        memory = get_long_memory()
        summaries = memory.get_summaries(
            session_id=session_id,
            limit=limit,
            min_importance=min_importance
        )
        return {
            "success": True,
            "count": len(summaries),
            "summaries": summaries
        }
    except Exception as e:
        logger.error(f"Get summaries error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/long-term/recent")
async def get_recent_summaries(count: int = 5):
    try:
        memory = get_long_memory()
        summaries = memory.get_recent_summaries(count)
        return {
            "success": True,
            "count": len(summaries),
            "summaries": summaries
        }
    except Exception as e:
        logger.error(f"Get recent summaries error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/long-term/delete/{summary_id}")
async def delete_summary(summary_id: int):
    try:
        memory = get_long_memory()
        success = memory.delete_summary(summary_id)
        if success:
            return {"success": True, "message": f"Summary {summary_id} deleted"}
        raise HTTPException(status_code=404, detail="Summary not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete summary error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_memory_stats():
    try:
        short_mem = get_short_memory()
        long_mem = get_long_memory()
        
        buffer = short_mem.get_recent_messages()
        stats = long_mem.get_stats()
        
        return {
            "short_term": {
                "session_id": short_mem.current_session_id,
                "buffer_size": len(buffer),
                "max_size": short_mem.max_size,
                "storage_type": type(short_mem.storage).__name__
            },
            "long_term": stats
        }
    except Exception as e:
        logger.error(f"Get memory stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
