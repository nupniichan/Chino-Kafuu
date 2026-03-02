import logging
import asyncio
import time
import uuid
from typing import Optional, Dict, Any, List

from src.core.event_bus import EventBus, Priority
from src.core import events
from src.modules.memory.memory_manager import MemoryManager
from src.modules.dialog.prompt_builder import PromptBuilder
from src.modules.dialog.llm_wrapper import BaseLLMWrapper

logger = logging.getLogger(__name__)


class DialogOrchestrator:
    """Event-driven dialog engine: memory query -> prompt -> LLM -> publish response."""

    def __init__(
        self,
        event_bus: EventBus,
        llm_wrapper: BaseLLMWrapper,
        memory_manager: MemoryManager,
        idle_timeout: int = 30,
    ):
        self.bus = event_bus
        self.llm = llm_wrapper
        self.memory = memory_manager
        self.prompt_builder = PromptBuilder()

        self.idle_timeout = idle_timeout
        self.last_interaction_time = time.time()
        self.is_processing = False
        self._cancelled = False
        self._lock = asyncio.Lock()
        self.auto_trigger_task: Optional[asyncio.Task] = None

    def register(self) -> None:
        self.bus.subscribe(
            events.STT_READY, self._on_stt_ready, owner="DialogEngine"
        )
        self.bus.subscribe(
            events.INTERRUPT, self._on_interrupt, priority=Priority.HIGH, owner="DialogEngine"
        )
        logger.info("DialogEngine registered on EventBus")

    # ---- Event handlers ----

    async def _on_stt_ready(self, event: str, data: events.STTReadyPayload) -> None:
        await self.process_user_message(
            user_message=data.text,
            user_emotion=data.emotion,
            user_lang=data.lang,
            source=data.source,
        )

    async def _on_interrupt(self, event: str, data: events.InterruptPayload) -> None:
        logger.info(f"DialogEngine received INTERRUPT: {data.reason}")
        self._cancelled = True

    # ---- Core pipeline ----

    async def process_user_message(
        self,
        user_message: str,
        user_emotion: str = "normal",
        user_lang: str = "en",
        source: str = "mic",
    ) -> List[Dict[str, Any]]:
        """Run pipeline: get memory -> build prompt -> LLM -> publish response."""
        async with self._lock:
            self.is_processing = True
            self._cancelled = False
            self.last_interaction_time = time.time()

            try:
                long_term_summaries = self.memory.get_recent_summaries(limit=3)
                conversation_history = self.memory.get_conversation_context()

                messages = self.prompt_builder.build_prompt(
                    user_message=user_message,
                    conversation_history=conversation_history,
                    is_auto_trigger=False,
                )

                if long_term_summaries:
                    messages.insert(1, {"role": "system", "content": long_term_summaries})

                if self._cancelled:
                    logger.info("DialogEngine cancelled before LLM call")
                    return []

                start_time = time.time()
                response_sentences = await asyncio.to_thread(
                    self.llm.generate_and_parse, messages
                )
                latency_ms = int((time.time() - start_time) * 1000)

                if self._cancelled:
                    logger.info("DialogEngine cancelled after LLM call")
                    return []

                logger.info(f"LLM response: {len(response_sentences)} sentences in {latency_ms}ms")

                response_id = str(uuid.uuid4())
                payload = events.LLMResponsePayload(
                    sentences=response_sentences,
                    response_id=response_id,
                    latency_ms=latency_ms,
                )
                await self.bus.publish(events.LLM_RESPONSE, payload)

                return response_sentences

            except Exception as e:
                logger.error(f"Error processing user message: {e}")
                raise
            finally:
                self.is_processing = False

    async def auto_trigger_conversation(self) -> List[Dict[str, Any]]:
        """Generate conversation when idle (no user input for idle_timeout seconds)."""
        if self._lock.locked():
            return []

        async with self._lock:
            self.is_processing = True
            self._cancelled = False

            try:
                logger.info("Auto-trigger: initiating conversation")

                long_term_summaries = self.memory.get_recent_summaries(limit=3)
                conversation_history = self.memory.get_conversation_context()

                messages = self.prompt_builder.build_prompt(
                    user_message=None,
                    conversation_history=conversation_history,
                    is_auto_trigger=True,
                )

                if long_term_summaries:
                    messages.insert(1, {"role": "system", "content": long_term_summaries})

                start_time = time.time()
                response_sentences = await asyncio.to_thread(
                    self.llm.generate_and_parse, messages
                )
                latency_ms = int((time.time() - start_time) * 1000)

                if self._cancelled:
                    return []

                payload = events.LLMResponsePayload(
                    sentences=response_sentences,
                    response_id=str(uuid.uuid4()),
                    latency_ms=latency_ms,
                    is_auto_trigger=True,
                )
                await self.bus.publish(events.LLM_RESPONSE, payload)

                self.last_interaction_time = time.time()
                return response_sentences

            except Exception as e:
                logger.error(f"Error in auto-trigger: {e}")
                return []
            finally:
                self.is_processing = False

    async def start_auto_trigger_loop(self) -> None:
        logger.info(f"Starting auto-trigger loop (timeout: {self.idle_timeout}s)")
        while True:
            try:
                await asyncio.sleep(5)
                if self.is_processing:
                    continue
                idle_duration = time.time() - self.last_interaction_time
                if idle_duration >= self.idle_timeout:
                    await self.auto_trigger_conversation()
            except asyncio.CancelledError:
                logger.info("Auto-trigger loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in auto-trigger loop: {e}")
                await asyncio.sleep(5)

    def start_auto_trigger(self) -> None:
        if self.auto_trigger_task is None or self.auto_trigger_task.done():
            self.auto_trigger_task = asyncio.create_task(self.start_auto_trigger_loop())
            logger.info("Auto-trigger task started")

    def stop_auto_trigger(self) -> None:
        if self.auto_trigger_task and not self.auto_trigger_task.done():
            self.auto_trigger_task.cancel()
            logger.info("Auto-trigger task stopped")

    # ---- API-compatible helpers ----

    def get_conversation_history(self, count: Optional[int] = None) -> List[Dict[str, Any]]:
        return self.memory.get_recent_messages(count)

    def clear_conversation(self) -> None:
        self.memory.clear()
        self.last_interaction_time = time.time()
        logger.info("Conversation cleared")

    def get_memory_stats(self) -> Dict[str, Any]:
        return self.memory.get_stats()
