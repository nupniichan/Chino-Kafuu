"""
Bootstrap: Centralized service registry and lifecycle management.

Creates all module instances, wires EventBus subscriptions,
and provides get_service() for API routes. Replaces scattered singletons.
"""
import logging
from typing import Any, Dict, Optional

from src.core.event_bus import EventBus
from src.core import events
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
    MEMORY_CACHE,
    REDIS_HOST,
    REDIS_PORT,
    REDIS_DB,
    IDLE_TIMEOUT_SECONDS,
    STT_MODEL_PATH,
    VAD_THRESHOLD,
    SILENCE_CHUNKS_NEEDED,
    SAMPLE_RATE,
)

logger = logging.getLogger(__name__)


class ServiceRegistry:
    """Single source of truth for all service instances."""

    def __init__(self) -> None:
        self._services: Dict[str, Any] = {}
        self._started = False

    @property
    def event_bus(self) -> EventBus:
        return self._services["event_bus"]

    def get(self, name: str) -> Any:
        if name not in self._services:
            raise KeyError(f"Service '{name}' not registered. Available: {list(self._services.keys())}")
        return self._services[name]

    async def startup(self, llm_mode: Optional[str] = None) -> None:
        if self._started:
            logger.warning("ServiceRegistry already started")
            return

        mode = llm_mode or LLM_MODE or "openrouter"
        logger.info(f"Bootstrapping services (llm_mode={mode})...")

        bus = EventBus()
        self._services["event_bus"] = bus

        self._init_memory(bus)
        self._init_dialog_engine(bus, mode)
        self._init_token_router(bus)

        logger.info(f"Bootstrap complete. Services: {list(self._services.keys())}")
        logger.info(f"EventBus stats: {bus.get_stats()}")
        self._started = True

    def _init_memory(self, bus: EventBus) -> None:
        from src.modules.memory.short_term import ShortTermMemory
        from src.modules.memory.long_term import LongTermMemory
        from src.modules.memory.summarizer import ConversationSummarizer
        from src.modules.memory.memory_manager import MemoryManager

        short_term = ShortTermMemory(
            max_size=SHORT_TERM_MEMORY_SIZE,
            storage_type=MEMORY_CACHE,
            redis_host=REDIS_HOST,
            redis_port=REDIS_PORT,
            redis_db=REDIS_DB,
        )
        long_term = LongTermMemory()

        self._services["short_term_memory"] = short_term
        self._services["long_term_memory"] = long_term

        llm_for_summary = self._create_llm("openrouter")
        summarizer = ConversationSummarizer(llm_for_summary)

        memory_manager = MemoryManager(
            event_bus=bus,
            short_term=short_term,
            long_term=long_term,
            summarizer=summarizer,
            compress_threshold=50,
        )
        memory_manager.register()
        self._services["memory_manager"] = memory_manager

    def _init_dialog_engine(self, bus: EventBus, mode: str) -> None:
        from src.modules.dialog.orchestrator import DialogOrchestrator

        llm = self._create_llm(mode)
        memory_manager = self._services["memory_manager"]

        engine = DialogOrchestrator(
            event_bus=bus,
            llm_wrapper=llm,
            memory_manager=memory_manager,
            idle_timeout=IDLE_TIMEOUT_SECONDS,
        )
        engine.register()
        self._services["dialog_engine"] = engine

    def _init_token_router(self, bus: EventBus) -> None:
        from src.modules.dialog.token_router import TokenRouter

        router = TokenRouter(num_slots=2)
        bus.subscribe(
            events.LLM_RESPONSE,
            self._make_token_router_handler(router),
            owner="TokenRouter",
        )
        bus.subscribe(
            events.INTERRUPT,
            self._make_token_router_interrupt_handler(router),
            owner="TokenRouter",
        )
        self._services["token_router"] = router

    @staticmethod
    def _make_token_router_handler(router):
        async def _on_llm_response(event: str, data: events.LLMResponsePayload) -> None:
            if data.sentences:
                await router.route_sentences(data.sentences, response_id=data.response_id)
                await router.process_all_sequential()
                logger.debug(f"TokenRouter processed {len(data.sentences)} sentences")
        return _on_llm_response

    @staticmethod
    def _make_token_router_interrupt_handler(router):
        async def _on_interrupt(event: str, data: events.InterruptPayload) -> None:
            router.reset()
            logger.info("TokenRouter reset due to interrupt")
        return _on_interrupt

    def _create_llm(self, mode: str):
        from src.modules.dialog.llm_wrapper import LocalLLMWrapper, OpenRouterLLMWrapper

        if mode == "local":
            return LocalLLMWrapper(
                model_path=LLM_MODEL_PATH,
                n_ctx=LLM_N_CTX,
                n_gpu_layers=LLM_N_GPU_LAYERS,
                temperature=LLM_TEMPERATURE,
                top_p=LLM_TOP_P,
                max_tokens=LLM_MAX_TOKENS,
            )
        elif mode == "openrouter":
            return OpenRouterLLMWrapper(
                api_key=OPENROUTER_API_KEY,
                model=OPENROUTER_MODEL,
                base_url=OPENROUTER_BASE_URL,
                timeout=OPENROUTER_TIMEOUT,
                temperature=LLM_TEMPERATURE,
                top_p=LLM_TOP_P,
                max_tokens=LLM_MAX_TOKENS,
            )
        else:
            raise ValueError(f"Invalid LLM mode: {mode}")

    async def shutdown(self) -> None:
        if not self._started:
            return

        logger.info("Shutting down services...")

        engine = self._services.get("dialog_engine")
        if engine:
            engine.stop_auto_trigger()

        pipeline = self._services.get("pipeline")
        if pipeline:
            pipeline.stop()

        bus = self._services.get("event_bus")
        if bus:
            await bus.publish(events.SYSTEM_SHUTDOWN, None)
            bus.reset()

        self._services.clear()
        self._started = False
        logger.info("Shutdown complete")


registry = ServiceRegistry()


def get_service(name: str) -> Any:
    return registry.get(name)
