from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# --- Event type constants ---

STT_READY = "stt_ready"
LLM_RESPONSE = "llm_response"
USER_MESSAGE = "user_message"
INTERRUPT = "interrupt"
MEMORY_FULL = "memory_full"
MEMORY_SAVED = "memory_saved"
SYSTEM_SHUTDOWN = "system_shutdown"


# --- Payload dataclasses ---

@dataclass
class STTReadyPayload:
    text: str
    lang: str = "en"
    source: str = "mic"
    emotion: str = "normal"


@dataclass
class LLMResponsePayload:
    sentences: List[Dict[str, Any]] = field(default_factory=list)
    response_id: str = ""
    latency_ms: int = 0
    is_auto_trigger: bool = False


@dataclass
class UserMessagePayload:
    text: str
    emotion: str = "normal"
    lang: str = "en"
    source: str = "text"
    interrupt: bool = False


@dataclass
class InterruptPayload:
    reason: str = "user_speaking"
    source: str = ""


@dataclass
class MemoryFullPayload:
    token_count: int = 0
    message_count: int = 0
    threshold: int = 8192
