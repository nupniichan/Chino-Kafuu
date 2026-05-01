"""
Memory API routes: Thin wrappers using bootstrap services.
"""
import logging
from fastapi import APIRouter, HTTPException, Query, Path
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

from src.core.bootstrap import get_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/memory", tags=["Memory"])


class MessageEntry(BaseModel):
    message: str = Field(..., min_length=1, examples=["Hello!"])
    emotion: str = Field(default="normal", examples=["normal", "happy", "sad"])
    lang: str = Field(default="en", pattern="^[a-z]{2}$")
    source: str = Field(default="api", examples=["api", "voice", "text"])


class ResponseEntry(BaseModel):
    text: str = Field(..., min_length=1)
    emotion: str = Field(default="normal")
    lang: str = Field(default="en", pattern="^[a-z]{2}$")


def _get_memory_manager():
    return get_service("memory_manager")


def _get_long_term():
    return get_service("long_term_memory")


@router.post("/short-term/add-user", summary="Add user message to short-term memory")
async def add_user_message(entry: MessageEntry):
    try:
        mm = _get_memory_manager()
        result = mm.short_term.add_user_message(
            message=entry.message,
            emotion=entry.emotion,
            lang=entry.lang,
            source=entry.source,
        )
        return {"success": True, "entry": result}
    except Exception as e:
        logger.error(f"Add user message error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/short-term/add-chino", summary="Add Chino's response to short-term memory")
async def add_chino_response(entry: ResponseEntry):
    try:
        mm = _get_memory_manager()
        result = mm.short_term.add_chino_response(
            text_spoken=entry.text,
            text_display=entry.text,
            emotion=entry.emotion,
            lang=entry.lang,
        )
        return {"success": True, "entry": result}
    except Exception as e:
        logger.error(f"Add chino response error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/short-term/buffer", summary="Get short-term memory buffer")
async def get_buffer():
    try:
        mm = _get_memory_manager()
        buffer = mm.get_recent_messages()
        return {"success": True, "buffer_size": len(buffer), "buffer": buffer}
    except Exception as e:
        logger.error(f"Get buffer error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/short-term/clear", summary="Clear short-term memory buffer")
async def clear_buffer():
    try:
        mm = _get_memory_manager()
        mm.clear()
        return {"success": True, "message": "Buffer cleared"}
    except Exception as e:
        logger.error(f"Clear buffer error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/long-term/summaries", summary="Get long-term memory summaries")
async def get_summaries(
    limit: int = Query(10, ge=1, le=100),
    min_importance: float = Query(0.0, ge=0.0, le=1.0),
):
    try:
        ltm = _get_long_term()
        summaries = ltm.get_recent_summaries(limit=limit, min_importance=min_importance)
        return {"success": True, "count": len(summaries), "summaries": summaries}
    except Exception as e:
        logger.error(f"Get summaries error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/long-term/recent", summary="Get recent long-term memory summaries")
async def get_recent_summaries(count: int = Query(5, ge=1, le=50)):
    try:
        ltm = _get_long_term()
        summaries = ltm.get_recent_summaries(count)
        return {"success": True, "count": len(summaries), "summaries": summaries}
    except Exception as e:
        logger.error(f"Get recent summaries error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/long-term/delete/{summary_id}", summary="Delete a memory summary")
async def delete_summary(summary_id: int = Path(..., ge=1)):
    try:
        ltm = _get_long_term()
        success = ltm.delete_summary(summary_id)
        if success:
            return {"success": True, "message": f"Summary {summary_id} deleted"}
        raise HTTPException(status_code=404, detail="Summary not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete summary error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", summary="Get memory system statistics")
async def get_memory_stats():
    try:
        mm = _get_memory_manager()
        return mm.get_stats()
    except Exception as e:
        logger.error(f"Get memory stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
