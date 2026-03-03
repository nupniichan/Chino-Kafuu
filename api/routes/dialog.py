import asyncio
import logging
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

from src.core.bootstrap import get_service
from src.core import events
from src.setting import LLM_MODE

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/dialog", tags=["Dialog"])


class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        examples=["Konichiwa", "Chino ơi, em khỏe hok?"],
    )
    emotion: str = Field(default="normal", examples=["normal", "happy", "sad"])
    lang: str = Field(default="en", pattern="^[a-z]{2}$", examples=["en", "vi", "ja"])
    source: str = Field(default="text", examples=["text", "voice", "api", "mic"])
    memory_cache: Optional[str] = Field(
        default=None,
        description="Type of memory cache to use for conversation history",
        examples=["redis", "in-memory"],
    )
    llm_mode: Optional[str] = Field(
        default=None,
        description="LLM provider mode. If not specified, uses default from settings",
        examples=["local", "openrouter"],
    )


class ChatResponse(BaseModel):
    responses: List[Dict[str, Any]]


class ConversationHistory(BaseModel):
    messages: List[Dict[str, Any]]
    count: int = Field(ge=0)


@router.post("/chat", response_model=ChatResponse, summary="Send a chat message")
async def chat(request: ChatRequest):
    """
    Process a user message through the full EventBus pipeline.

    Flow: publish STT_READY -> MemoryManager saves user msg
    -> Dialog Engine processes -> publish LLM_RESPONSE
    -> MemoryManager saves AI response -> TokenRouter routes
    -> captured response returned to caller.

    Optional overrides:
    - **memory_cache**: 'redis' or 'in-memory'
    - **llm_mode**: 'local' or 'openrouter'
    """
    llm_mode = request.llm_mode or LLM_MODE or "openrouter"

    if llm_mode not in ("local", "openrouter"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid llm_mode: {llm_mode}. Must be 'local' or 'openrouter'",
        )

    if request.memory_cache and request.memory_cache not in ("redis", "in-memory"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid memory_cache: {request.memory_cache}. Must be 'redis' or 'in-memory'",
        )

    try:
        bus = get_service("event_bus")
        response_holder: List[events.LLMResponsePayload] = []

        async def _capture(event: str, data: events.LLMResponsePayload) -> None:
            response_holder.append(data)

        bus.subscribe(events.LLM_RESPONSE, _capture, owner="_api_chat_temp")

        try:
            payload = events.STTReadyPayload(
                text=request.message,
                lang=request.lang,
                source=request.source,
                emotion=request.emotion,
                llm_mode=llm_mode,
            )
            await bus.publish(events.STT_READY, payload)

            if response_holder:
                return ChatResponse(responses=response_holder[0].sentences)
            return ChatResponse(responses=[])
        finally:
            bus.unsubscribe(events.LLM_RESPONSE, _capture)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")


@router.post("/chat-event", summary="Send chat via EventBus (fire-and-forget)")
async def chat_event(request: ChatRequest):
    """
    Publish user message as STT_READY event and return immediately.
    Response is processed asynchronously via EventBus (for real-time/streaming mode).
    """
    try:
        llm_mode = request.llm_mode or LLM_MODE or "openrouter"
        bus = get_service("event_bus")
        payload = events.STTReadyPayload(
            text=request.message,
            lang=request.lang,
            source=request.source,
            emotion=request.emotion,
            llm_mode=llm_mode,
        )
        asyncio.create_task(bus.publish(events.STT_READY, payload))
        return {"status": "published", "message": "Processing asynchronously"}
    except Exception as e:
        logger.error(f"Chat event error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history", response_model=ConversationHistory, summary="Get conversation history")
async def get_history(
    count: Optional[int] = Query(None, ge=1, le=1000),
    llm_mode: Optional[str] = Query(
        None,
        description="LLM mode to get history from",
        examples=["local", "openrouter"],
    ),
):
    try:
        engine = get_service("dialog_engine")
        messages = engine.get_conversation_history(count)
        return ConversationHistory(messages=messages, count=len(messages))
    except KeyError:
        raise HTTPException(status_code=503, detail="Dialog system not initialized")
    except Exception as e:
        logger.error(f"History error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/clear", summary="Clear conversation history")
async def clear_conversation(
    llm_mode: Optional[str] = Query(
        None,
        description="LLM mode to clear conversation for",
        examples=["local", "openrouter"],
    ),
):
    try:
        engine = get_service("dialog_engine")
        engine.clear_conversation()
        return {"status": "success", "message": "Conversation cleared"}
    except KeyError:
        raise HTTPException(status_code=503, detail="Dialog system not initialized")
    except Exception as e:
        logger.error(f"Clear error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status", summary="Get dialog system status")
async def get_status(
    llm_mode: Optional[str] = Query(
        None,
        description="LLM mode to check status for",
        examples=["local", "openrouter"],
    ),
):
    try:
        engine = get_service("dialog_engine")
        bus = get_service("event_bus")
        return {
            "status": "active",
            "is_processing": engine.is_processing,
            "idle_timeout": engine.idle_timeout,
            "auto_trigger_active": engine.auto_trigger_task is not None,
            "event_bus": bus.get_stats(),
        }
    except KeyError:
        return {"status": "not_initialized"}


@router.post("/interrupt", summary="Send interrupt signal")
async def interrupt():
    """Publish INTERRUPT event to stop all processing."""
    try:
        bus = get_service("event_bus")
        count = await bus.publish(
            events.INTERRUPT,
            events.InterruptPayload(reason="api_request"),
        )
        return {"status": "interrupted", "handlers_notified": count}
    except Exception as e:
        logger.error(f"Interrupt error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/memory_stats", summary="Get memory statistics")
async def get_memory_stats(
    llm_mode: Optional[str] = Query(
        None,
        description="LLM mode to get memory stats for",
        examples=["local", "openrouter"],
    ),
):
    try:
        engine = get_service("dialog_engine")
        return engine.get_memory_stats()
    except KeyError:
        raise HTTPException(status_code=503, detail="Dialog system not initialized")
    except Exception as e:
        logger.error(f"Memory stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
