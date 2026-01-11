"""
Dialog API route: Exposes dialog system via REST API.
Provides endpoints for chat and conversation management.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/dialog", tags=["Dialog"])


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str
    emotion: str = "normal"
    lang: str = "en"
    source: str = "text"
    memory_cache: Optional[str] = None


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    responses: List[Dict[str, Any]]
    conversation_id: str


class ConversationHistory(BaseModel):
    """Response model for conversation history."""
    messages: List[Dict[str, Any]]
    count: int


orchestrator_instance = None


def set_orchestrator(orchestrator):
    """Set global orchestrator instance."""
    global orchestrator_instance
    orchestrator_instance = orchestrator


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process user message and return Chino's response.
    
    Args:
        request: Chat request with optional memory_backend ("redis" or "cache")
    """
    
    if orchestrator_instance is None:
        raise HTTPException(status_code=503, detail="Dialog system not initialized")
    
    if request.memory_cache and request.memory_cache not in ["redis", "cache"]:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid memory_cache: {request.memory_cache}. Must be 'redis' or 'cache'"
        )
    
    try:
        responses = await orchestrator_instance.process_user_message(
            user_message=request.message,
            user_emotion=request.emotion,
            user_lang=request.lang,
            source=request.source
        )
        
        session_id = orchestrator_instance.memory.current_session_id or "default"
        
        return ChatResponse(
            responses=responses,
            conversation_id=session_id
        )
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")


@router.get("/history", response_model=ConversationHistory)
async def get_history(count: Optional[int] = None):
    """Get conversation history."""
    
    if orchestrator_instance is None:
        raise HTTPException(status_code=503, detail="Dialog system not initialized")
    
    try:
        messages = orchestrator_instance.get_conversation_history(count)
        
        return ConversationHistory(
            messages=messages,
            count=len(messages)
        )
        
    except Exception as e:
        logger.error(f"History retrieval error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get history: {str(e)}")


@router.post("/clear")
async def clear_conversation():
    """Clear conversation history."""
    
    if orchestrator_instance is None:
        raise HTTPException(status_code=503, detail="Dialog system not initialized")
    
    try:
        orchestrator_instance.clear_conversation()
        return {"status": "success", "message": "Conversation cleared"}
        
    except Exception as e:
        logger.error(f"Clear conversation error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear: {str(e)}")


@router.get("/status")
async def get_status():
    """Get dialog system status."""
    
    if orchestrator_instance is None:
        return {"status": "not_initialized"}
    
    return {
        "status": "active",
        "is_processing": orchestrator_instance.is_processing,
        "messages_in_memory": len(orchestrator_instance.memory.buffer),
        "idle_timeout": orchestrator_instance.idle_timeout,
        "auto_trigger_active": orchestrator_instance.auto_trigger_task is not None
    }
