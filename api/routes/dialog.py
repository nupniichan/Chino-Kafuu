from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging

from src.modules.dialog.llm_wrapper import LocalLLMWrapper, OpenRouterLLMWrapper
from src.modules.dialog.orchestrator import DialogOrchestrator
from src.modules.memory.long_term import LongTermMemory
from api.routes.memory import get_short_memory
from src.setting import (
    LLM_MODE,
    LLM_MODEL_PATH,
    LLM_N_CTX,
    LLM_N_GPU_LAYERS,
    LLM_TEMPERATURE,
    LLM_TOP_P,
    LLM_MAX_TOKENS,
    OPENROUTER_API_KEY,
    OPENROUTER_MODEL,
    OPENROUTER_BASE_URL,
    OPENROUTER_TIMEOUT,
    SHORT_TERM_TOKEN_LIMIT,
    MEMORY_IMPORTANCE_THRESHOLD,
    IDLE_TIMEOUT_SECONDS,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/dialog", tags=["Dialog"])


class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        description="The user's message/question to the AI assistant",
        title="User Message",
        min_length=1,
        examples=["Konichiwa", "Chino ơi, em khỏe hok?", "You're smol and cute", "こんにちは！"]
    )
    
    emotion: str = Field(
        default="normal",
        description="Emotional state of the user when sending the message",
        title="User Emotion",
        examples=["normal", "happy", "sad", "angry", "excited", "confused", "..."]
    )
    
    lang: str = Field(
        default="en",
        description="Language code for the conversation",
        title="Language",
        examples=["en", "vi", "ja"],
        pattern="^[a-z]{2}$"
    )
    
    source: str = Field(
        default="text",
        description="Source/origin of the message",
        title="Message Source",
        examples=["text", "voice", "api", "mic"]
    )
    
    memory_cache: Optional[str] = Field(
        default=None,
        description="Type of memory cache to use for conversation history",
        title="Memory Cache Type",
        examples=["redis", "in-memory"]
    )
    
    llm_mode: Optional[str] = Field(
        default=None,
        description="LLM provider mode to use. If not specified, uses default from settings.py",
        title="LLM Mode",
        examples=["local", "openrouter"]
    )

    class Config:
        json_schema_extra = {
            "example": {
                "message": "Hello! How are you today?",
                "emotion": "happy",
                "lang": "en",
                "source": "text",
                "memory_cache": "in-memory",
                "llm_mode": "openrouter"
            }
        }


class ChatResponse(BaseModel):
    responses: List[Dict[str, Any]] = Field(
        ...,
        description="List of AI responses with metadata",
        title="Response List"
    )
class ConversationHistory(BaseModel):
    messages: List[Dict[str, Any]] = Field(
        ...,
        description="List of conversation messages in chronological order",
        title="Message History"
    )
    
    count: int = Field(
        ...,
        description="Total number of messages in the history",
        title="Message Count",
        ge=0
    )


orchestrator_instances: Dict[str, DialogOrchestrator] = {}


def _create_llm_instance(mode: str):
    if mode == "local":
        return LocalLLMWrapper(
            model_path=LLM_MODEL_PATH,
            n_ctx=LLM_N_CTX,
            n_gpu_layers=LLM_N_GPU_LAYERS,
            temperature=LLM_TEMPERATURE,
            top_p=LLM_TOP_P,
            max_tokens=LLM_MAX_TOKENS
        )
    elif mode == "openrouter":
        return OpenRouterLLMWrapper(
            api_key=OPENROUTER_API_KEY,
            model=OPENROUTER_MODEL,
            base_url=OPENROUTER_BASE_URL,
            timeout=OPENROUTER_TIMEOUT,
            temperature=LLM_TEMPERATURE,
            top_p=LLM_TOP_P,
            max_tokens=LLM_MAX_TOKENS
        )
    else:
        raise ValueError(f"Invalid LLM_MODE: {mode}. Must be 'local' or 'openrouter'")


def _get_orchestrator(mode: str) -> DialogOrchestrator:
    if mode not in orchestrator_instances:
        llm = _create_llm_instance(mode)
        short_memory = get_short_memory()
        long_memory = LongTermMemory()
        
        orchestrator_instances[mode] = DialogOrchestrator(
            llm_wrapper=llm,
            short_term_memory=short_memory,
            long_term_memory=long_memory,
            idle_timeout=IDLE_TIMEOUT_SECONDS,
            token_limit=SHORT_TERM_TOKEN_LIMIT,
            importance_threshold=MEMORY_IMPORTANCE_THRESHOLD
        )
        logger.info(f"Created orchestrator instance for mode: {mode}")
        logger.info(f"Token limit: {SHORT_TERM_TOKEN_LIMIT}, Importance threshold: {MEMORY_IMPORTANCE_THRESHOLD}")
    return orchestrator_instances[mode]


@router.post("/chat", response_model=ChatResponse, summary="Send a chat message")
async def chat(request: ChatRequest):
    """
    Process a user message and generate AI responses
    
    ## Parameters:
    - **message**: Your question or message to the AI (required)
    - **emotion**: Your emotional state when sending the message
    - **lang**: Language code (e.g., 'en', 'vi', 'ja')
    - **source**: Where the message is coming from
    - **memory_cache**: Type of cache to use for conversation history ('in-memory' or 'redis')
    - **llm_mode**: LLM provider ('local' or 'openrouter')
    
    ## Returns:
    - **responses**: List of AI-generated responses with metadata
    - **conversation_id**: ID to track this conversation session
    """
    llm_mode = request.llm_mode or LLM_MODE or "openrouter"
    
    if llm_mode not in ["local", "openrouter"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid llm_mode: {llm_mode}. Must be 'local' or 'openrouter'"
        )
    
    if request.memory_cache and request.memory_cache not in ["redis", "in-memory"]:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid memory_cache: {request.memory_cache}. Must be 'redis' or 'in-memory'"
        )
    
    try:
        orchestrator = _get_orchestrator(llm_mode)
        
        responses = await orchestrator.process_user_message(
            user_message=request.message,
            user_emotion=request.emotion,
            user_lang=request.lang,
            source=request.source
        )
        
        return ChatResponse(responses=responses)
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")


@router.get("/history", response_model=ConversationHistory, summary="Get conversation history")
async def get_history(
    count: Optional[int] = Query(
        None,
        description="Number of recent messages to retrieve. If not specified, returns all messages",
        title="Message Count",
        ge=1,
        le=1000
    ),
    llm_mode: Optional[str] = Query(
        None,
        description="LLM mode to get history from",
        title="LLM Mode",
        examples=["local", "openrouter"]
    )
):
    """
    Retrieve conversation history
    
    ## Parameters:
    - **count**: Number of recent messages to get (optional, returns all if not specified)
    - **llm_mode**: Which LLM mode's history to retrieve
    
    ## Returns:
    - **messages**: List of conversation messages
    - **count**: Total number of messages returned
    """
    mode = llm_mode or LLM_MODE or "openrouter"
    
    if mode not in orchestrator_instances:
        raise HTTPException(status_code=503, detail=f"Dialog system not initialized for mode: {mode}")
    
    try:
        orchestrator = orchestrator_instances[mode]
        messages = orchestrator.get_conversation_history(count)
        
        return ConversationHistory(
            messages=messages,
            count=len(messages)
        )
        
    except Exception as e:
        logger.error(f"History retrieval error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get history: {str(e)}")


@router.post("/clear", summary="Clear conversation history")
async def clear_conversation(
    llm_mode: Optional[str] = Query(
        None,
        description="LLM mode to clear conversation for",
        title="LLM Mode",
        examples=["local", "openrouter"]
    )
):
    """
    Clear the conversation history and start fresh
    
    ## Parameters:
    - **llm_mode**: Which LLM mode's conversation to clear
    
    ## Returns:
    - Success status and confirmation message
    """
    mode = llm_mode or LLM_MODE or "openrouter"
    
    if mode not in orchestrator_instances:
        raise HTTPException(status_code=503, detail=f"Dialog system not initialized for mode: {mode}")
    
    try:
        orchestrator = orchestrator_instances[mode]
        orchestrator.clear_conversation()
        return {"status": "success", "message": "Conversation cleared"}
        
    except Exception as e:
        logger.error(f"Clear conversation error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear: {str(e)}")


@router.get("/status", summary="Get dialog system status")
async def get_status(
    llm_mode: Optional[str] = Query(
        None,
        description="LLM mode to check status for",
        title="LLM Mode",
        examples=["local", "openrouter"]
    )
):
    """
    Check the current status of the dialog system
    
    ## Parameters:
    - **llm_mode**: Which LLM mode to check
    
    ## Returns:
    - **status**: System status (active/not_initialized)
    - **mode**: Current LLM mode
    - **is_processing**: Whether system is currently processing a message
    - **messages_in_memory**: Number of messages in short-term memory
    - **idle_timeout**: Seconds before conversation times out
    - **auto_trigger_active**: Whether auto-trigger is running
    - **available_modes**: List of initialized LLM modes
    """
    mode = llm_mode or LLM_MODE or "openrouter"
    
    if mode not in orchestrator_instances:
        return {
            "status": "not_initialized",
            "mode": mode,
            "available_modes": list(orchestrator_instances.keys())
        }
    
    orchestrator = orchestrator_instances[mode]
    return {
        "status": "active",
        "mode": mode,
        "is_processing": orchestrator.is_processing,
        "messages_in_memory": len(orchestrator.short_memory.buffer),
        "idle_timeout": orchestrator.idle_timeout,
        "auto_trigger_active": orchestrator.auto_trigger_task is not None,
        "available_modes": list(orchestrator_instances.keys())
    }


@router.get("/memory_stats", summary="Get memory statistics")
async def get_memory_stats(
    llm_mode: Optional[str] = Query(
        None,
        description="LLM mode to get memory stats for",
        title="LLM Mode",
        examples=["local", "openrouter"]
    )
):
    """
    Get detailed statistics about memory usage
    
    ## Parameters:
    - **llm_mode**: Which LLM mode's memory to check
    
    ## Returns:
    - Detailed memory statistics including:
      - Short-term memory usage
      - Long-term memory entries
      - Token counts
      - Importance scores
    """
    mode = llm_mode or LLM_MODE or "openrouter"
    
    if mode not in orchestrator_instances:
        raise HTTPException(status_code=503, detail=f"Dialog system not initialized for mode: {mode}")
    
    try:
        orchestrator = orchestrator_instances[mode]
        stats = orchestrator.get_memory_stats()
        
        return stats
        
    except Exception as e:
        logger.error(f"Memory stats error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")