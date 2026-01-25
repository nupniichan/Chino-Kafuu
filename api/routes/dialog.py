from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging

from src.modules.dialog.llm_wrapper import LocalLLMWrapper, OpenRouterLLMWrapper
from src.modules.dialog.orchestrator import DialogOrchestrator
from src.modules.memory.short_term import ShortTermMemory
from src.modules.memory.long_term import LongTermMemory
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
    SHORT_TERM_MEMORY_SIZE,
    SHORT_TERM_TOKEN_LIMIT,
    MEMORY_IMPORTANCE_THRESHOLD,
    IDLE_TIMEOUT_SECONDS,
    MEMORY_CACHE,
    REDIS_HOST,
    REDIS_PORT,
    REDIS_DB
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/dialog", tags=["Dialog"])


class ChatRequest(BaseModel):
    message: str
    emotion: str = "normal"
    lang: str = "en"
    source: str = "text"
    memory_cache: Optional[str] = None
    llm_mode: Optional[str] = None


class ChatResponse(BaseModel):
    responses: List[Dict[str, Any]]
    conversation_id: str


class ConversationHistory(BaseModel):
    messages: List[Dict[str, Any]]
    count: int


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
        short_memory = ShortTermMemory(
            max_size=SHORT_TERM_MEMORY_SIZE,
            storage_type=MEMORY_CACHE,
            redis_host=REDIS_HOST,
            redis_port=REDIS_PORT,
            redis_db=REDIS_DB
        )
        
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


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    llm_mode = request.llm_mode or LLM_MODE or "openrouter"
    
    if llm_mode not in ["local", "openrouter"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid llm_mode: {llm_mode}. Must be 'local' or 'openrouter'"
        )
    
    if request.memory_cache and request.memory_cache not in ["redis", "in-memory"]:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid memory_cache: {request.memory_cache}. Must be 'redis', 'cache', or 'in-memory'"
        )
    
    try:
        orchestrator = _get_orchestrator(llm_mode)
        
        responses = await orchestrator.process_user_message(
            user_message=request.message,
            user_emotion=request.emotion,
            user_lang=request.lang,
            source=request.source
        )
        
        session_id = orchestrator.short_memory.current_session_id or "default"
        
        return ChatResponse(
            responses=responses,
            conversation_id=session_id
        )
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")


@router.get("/history", response_model=ConversationHistory)
async def get_history(count: Optional[int] = None, llm_mode: Optional[str] = None):
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


@router.post("/clear")
async def clear_conversation(llm_mode: Optional[str] = None):
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


@router.get("/status")
async def get_status(llm_mode: Optional[str] = None):
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


@router.get("/memory_stats")
async def get_memory_stats(llm_mode: Optional[str] = None):
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
