"""
Memory Manager: Event-driven facade over ShortTerm, LongTerm, and Summarizer.

Subscribes to EventBus for automatic save. Exposes direct-call query methods
for Dialog Engine (which needs return values).
"""
import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from src.core.event_bus import EventBus, Priority
from src.core import events
from src.modules.memory.short_term import ShortTermMemory
from src.modules.memory.long_term import LongTermMemory
from src.modules.memory.summarizer import ConversationSummarizer

logger = logging.getLogger(__name__)


class MemoryManager:
    def __init__(
        self,
        event_bus: EventBus,
        short_term: ShortTermMemory,
        long_term: Optional[LongTermMemory] = None,
        summarizer: Optional[ConversationSummarizer] = None,
        compress_threshold: int = 50,
    ):
        self.bus = event_bus
        self.short_term = short_term
        self.long_term = long_term
        self.summarizer = summarizer
        self.compress_threshold = compress_threshold
        self._lock = asyncio.Lock()

    def register(self) -> None:
        self.bus.subscribe(events.STT_READY, self._on_stt_ready, owner="MemoryManager")
        self.bus.subscribe(events.LLM_RESPONSE, self._on_llm_response, owner="MemoryManager")
        logger.info("MemoryManager registered on EventBus")

    # ---- Event handlers (fire-and-forget saves) ----

    async def _on_stt_ready(self, event: str, data: events.STTReadyPayload) -> None:
        try:
            self.short_term.add_user_message(
                message=data.text,
                emotion=data.emotion,
                lang=data.lang,
                source=data.source,
            )
            logger.debug(f"MemoryManager saved user message: {data.text[:50]}...")
        except Exception as e:
            logger.error(f"MemoryManager failed to save user message: {e}")

    async def _on_llm_response(self, event: str, data: events.LLMResponsePayload) -> None:
        try:
            for idx, sentence in enumerate(data.sentences):
                is_last = idx == len(data.sentences) - 1
                self.short_term.add_chino_response(
                    text_spoken=sentence.get("text_spoken", ""),
                    text_display=sentence.get("text_display", ""),
                    lang="jp",
                    emotion=sentence.get("emo", "normal"),
                    action=sentence.get("act", "none"),
                    intensity=sentence.get("intensity", 0.5),
                    stream_index=idx,
                    is_completed=is_last,
                    latency_ms=data.latency_ms if idx == 0 else 0,
                )
            logger.debug(f"MemoryManager saved {len(data.sentences)} response sentences")
            await self._check_and_compress()
        except Exception as e:
            logger.error(f"MemoryManager failed to save LLM response: {e}")

    async def _check_and_compress(self) -> None:
        messages = self.short_term.get_recent_messages()
        if len(messages) < self.compress_threshold:
            return

        logger.info(f"Memory threshold reached ({len(messages)} >= {self.compress_threshold}), compressing")

        if not self.long_term or not self.summarizer:
            logger.warning("Long-term memory or summarizer not available, skipping compression")
            return

        try:
            async with self._lock:
                summary = await asyncio.to_thread(
                    self.summarizer.summarize_conversation, messages
                )
                importance = self.summarizer.calculate_importance_score(messages, summary)

                self.long_term.add_summary(
                    summary=summary,
                    original_messages=messages,
                    importance_score=importance,
                    metadata={
                        "compressed_at": int(time.time() * 1000),
                        "message_count": len(messages),
                    },
                )

                self.short_term.storage.trim(
                    self.short_term.storage_key, len(messages), -1
                )

                logger.info(f"Compressed {len(messages)} messages (importance={importance:.2f})")

                await self.bus.publish(
                    events.MEMORY_FULL,
                    events.MemoryFullPayload(
                        message_count=len(messages),
                        threshold=self.compress_threshold,
                    ),
                )
        except Exception as e:
            logger.error(f"Memory compression failed: {e}")

    # ---- Direct-call query methods (for Dialog Engine) ----

    def get_conversation_context(self) -> List[Dict[str, str]]:
        return self.short_term.get_conversation_context()

    def get_recent_summaries(self, limit: int = 3) -> str:
        if not self.long_term:
            return ""
        try:
            summaries = self.long_term.get_recent_summaries(limit=limit)
            if summaries:
                lines = [f"- {s['summary']}" for s in summaries]
                return f"Recent conversation summaries:\n" + "\n".join(lines)
        except Exception as e:
            logger.error(f"Failed to retrieve summaries: {e}")
        return ""

    def get_recent_messages(self, count: Optional[int] = None) -> List[Dict[str, Any]]:
        return self.short_term.get_recent_messages(count)

    def clear(self) -> None:
        self.short_term.clear()
        logger.info("Memory cleared")

    def get_stats(self) -> Dict[str, Any]:
        stats: Dict[str, Any] = {
            "short_term": {"messages": len(self.short_term.buffer)},
        }
        if self.long_term:
            try:
                stats["long_term"] = self.long_term.get_stats()
            except Exception:
                stats["long_term"] = {"status": "error"}
        return stats
