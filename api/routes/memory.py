import logging
from fastapi import APIRouter, HTTPException, Query, Path
from pydantic import BaseModel, Field
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
    message: str = Field(
        ...,
        description="The user's message content",
        title="Message",
        min_length=1,
        examples=["Hello!", "Em khoẻ hok?", "こんにちは"]
    )
    
    emotion: str = Field(
        default="normal",
        description="Emotional state when the message was sent",
        title="Emotion",
        examples=["normal", "happy", "sad", "angry", "excited"]
    )
    
    lang: str = Field(
        default="en",
        description="Language code of the message",
        title="Language",
        examples=["en", "vi", "ja"],
        pattern="^[a-z]{2}$"
    )
    
    source: str = Field(
        default="api",
        description="Source/origin of the message",
        title="Message Source",
        examples=["api", "voice", "text"]
    )

    class Config:
        json_schema_extra = {
            "example": {
                "message": "Hello! How are you today?",
                "emotion": "happy",
                "lang": "en",
                "source": "api"
            }
        }


class ResponseEntry(BaseModel):
    text: str = Field(
        ...,
        description="Chino's response text",
        title="Response Text",
        min_length=1,
        examples=["Hello! I'm doing well, thank you!", "こんにちは！元気ですよ"]
    )
    
    emotion: str = Field(
        default="normal",
        description="Emotional state in the response",
        title="Emotion",
        examples=["normal", "happy", "sad", "playful", "shy"]
    )
    
    lang: str = Field(
        default="en",
        description="Language code of the response (ISO 639-1)",
        title="Language",
        examples=["en", "vi", "ja", "zh"],
        pattern="^[a-z]{2}$"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "text": "I'm happy to help you!",
                "emotion": "cheerful",
                "lang": "en"
            }
        }


@router.post("/short-term/add-user", summary="Add user message to short-term memory")
async def add_user_message(entry: MessageEntry):
    """
    Add a user message to the short-term memory buffer
    
    ## Parameters:
    - **message**: The user's message text (required)
    - **emotion**: Emotional state of the user
    - **lang**: Language code (e.g., 'en', 'ja', 'vi')
    - **source**: Where the message came from
    
    ## Returns:
    - **success**: Boolean indicating success
    - **entry**: The stored memory entry with metadata
    """
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


@router.post("/short-term/add-chino", summary="Add Chino's response to short-term memory")
async def add_chino_response(entry: ResponseEntry):
    """
    Add Chino's response to the short-term memory buffer
    
    ## Parameters:
    - **text**: Chino's response text (required)
    - **emotion**: Emotional state in the response
    - **lang**: Language code of the response
    
    ## Returns:
    - **success**: Boolean indicating success
    - **entry**: The stored memory entry with metadata
    """
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


@router.get("/short-term/buffer", summary="Get short-term memory buffer")
async def get_buffer():
    """
    Retrieve all messages currently in the short-term memory buffer
    
    ## Returns:
    - **success**: Boolean indicating success
    - **session_id**: Current conversation session ID
    - **buffer_size**: Number of messages in buffer
    - **buffer**: List of all messages in chronological order
    """
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


@router.post("/short-term/clear", summary="Clear short-term memory buffer")
async def clear_buffer():
    """
    Clear all messages from the short-term memory buffer
    
    ## Returns:
    - **success**: Boolean indicating success
    - **message**: Confirmation message
    """
    try:
        memory = get_short_memory()
        memory.clear()
        return {"success": True, "message": "Buffer cleared"}
    except Exception as e:
        logger.error(f"Clear buffer error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/short-term/new-session", summary="Start a new conversation session")
async def start_new_session():
    """
    Start a new conversation session with a fresh session ID
    
    ## Returns:
    - **success**: Boolean indicating success
    - **session_id**: The newly created session ID
    """
    try:
        memory = get_short_memory()
        session_id = memory.start_new_session()
        return {"success": True, "session_id": session_id}
    except Exception as e:
        logger.error(f"Start new session error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/long-term/summaries", summary="Get long-term memory summaries")
async def get_summaries(
    session_id: Optional[str] = Query(
        None,
        description="Filter by specific session ID. If not provided, returns summaries from all sessions",
        title="Session ID",
        examples=["session_123", "user_abc_20250129"]
    ),
    limit: int = Query(
        10,
        description="Maximum number of summaries to return",
        title="Result Limit",
        ge=1,
        le=100
    ),
    min_importance: float = Query(
        0.0,
        description="Minimum importance score (0.0 to 1.0). Only summaries with importance >= this value will be returned",
        title="Minimum Importance Score",
        ge=0.0,
        le=1.0
    )
):
    """
    Retrieve conversation summaries from long-term memory
    
    ## Parameters:
    - **session_id**: Filter by session (optional, returns all if not specified)
    - **limit**: Maximum number of results (1-100)
    - **min_importance**: Minimum importance threshold (0.0-1.0)
    
    ## Returns:
    - **success**: Boolean indicating success
    - **count**: Number of summaries returned
    - **summaries**: List of memory summaries with metadata
    """
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


@router.get("/long-term/recent", summary="Get recent long-term memory summaries")
async def get_recent_summaries(
    count: int = Query(
        5,
        description="Number of most recent summaries to retrieve",
        title="Summary Count",
        ge=1,
        le=50
    )
):
    """
    Get the most recent conversation summaries from long-term memory
    
    ## Parameters:
    - **count**: Number of recent summaries to retrieve (1-50)
    
    ## Returns:
    - **success**: Boolean indicating success
    - **count**: Number of summaries returned
    - **summaries**: List of recent summaries ordered by recency
    """
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


@router.delete("/long-term/delete/{summary_id}", summary="Delete a memory summary")
async def delete_summary(
    summary_id: int = Path(
        ...,
        description="ID of the summary to delete",
        title="Summary ID",
        ge=1
    )
):
    """
    Delete a specific summary from long-term memory
    
    ## Parameters:
    - **summary_id**: The ID of the summary to delete (required)
    
    ## Returns:
    - **success**: Boolean indicating success
    - **message**: Confirmation message
    
    ## Errors:
    - 404: Summary not found
    - 500: Server error
    """
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


@router.get("/stats", summary="Get memory system statistics")
async def get_memory_stats():
    """
    Get comprehensive statistics about both short-term and long-term memory
    
    ## Returns:
    - **short_term**: Statistics about short-term memory
      - session_id: Current session ID
      - buffer_size: Number of messages in buffer
      - max_size: Maximum buffer capacity
      - storage_type: Type of storage backend used
    - **long_term**: Statistics about long-term memory
      - Total summaries count
      - Average importance scores
      - Storage information
    """
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