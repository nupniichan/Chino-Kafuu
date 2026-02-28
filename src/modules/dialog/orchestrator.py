"""
Dialog orchestrator: Manages conversation flow with LLM, memory, and auto-trigger.
Coordinates between user input, memory retrieval, prompt building, and LLM response.
"""
import logging
import asyncio
from typing import Optional, Dict, Any, List
from datetime import datetime
import time

from src.modules.memory.short_term import ShortTermMemory
from src.modules.memory.long_term import LongTermMemory
from src.modules.memory.summarizer import ConversationSummarizer
from src.modules.dialog.prompt_builder import PromptBuilder
from src.modules.dialog.llm_wrapper import BaseLLMWrapper

logger = logging.getLogger(__name__)


class DialogOrchestrator:
    """Orchestrates dialog flow: memory → prompt → LLM → response → memory."""
    
    def __init__(
        self,
        llm_wrapper: BaseLLMWrapper,
        short_term_memory: Optional[ShortTermMemory] = None,
        long_term_memory: Optional[LongTermMemory] = None,
        idle_timeout: int = 30,
        token_limit: int = 8000,
        importance_threshold: float = 0.8
    ):
        """Initialize with LLM, all memory layers, and settings."""
        self.llm = llm_wrapper
        self.short_memory = short_term_memory or ShortTermMemory()
        self.long_memory = long_term_memory
        
        self.summarizer = ConversationSummarizer(llm_wrapper)
        self.prompt_builder = PromptBuilder()
        
        self.idle_timeout = idle_timeout
        self.token_limit = token_limit
        self.importance_threshold = importance_threshold
        
        self.last_interaction_time = time.time()
        self.is_processing = False
        self._lock = asyncio.Lock()
        self.auto_trigger_task: Optional[asyncio.Task] = None
    
    
    async def _retrieve_long_term_summaries(self) -> str:
        """Retrieve recent long-term conversation summaries."""
        if not self.long_memory:
            return ""
        
        try:
            summaries = self.long_memory.get_recent_summaries(limit=3)
            
            if summaries:
                summary_text = "\n".join([
                    f"- {s['summary']}" for s in summaries
                ])
                return f"Recent conversation summaries:\n{summary_text}"
            
        except Exception as e:
            logger.error(f"Failed to retrieve long-term summaries: {e}")
        
        return ""
    
    async def _check_and_compress_memory(self):
        """Check short-term memory size and compress if needed."""
        recent_messages = self.short_memory.get_recent_messages()
        message_count = len(recent_messages)
        
        logger.info(f"Memory Stats - Short-term: {message_count} messages")
        
        # Compress when message count reaches a threshold (e.g., 50 messages)
        if message_count >= 50:
            logger.info(f"Message limit reached ({message_count} messages), compressing memory")
            await self._compress_to_long_term(recent_messages)
    
    async def _compress_to_long_term(self, messages: List[Dict[str, Any]]):
        """Compress short-term to long-term memory via summarization."""
        if not self.long_memory:
            logger.warning("Long-term memory not initialized, skipping compression")
            return
        
        try:
            summary = self.summarizer.summarize_conversation(messages)
            importance_score = self.summarizer.calculate_importance_score(messages, summary)
            
            message_count = len(messages)
            session_id = self.short_memory.current_session_id or "default"
            
            summary_id = self.long_memory.add_summary(
                session_id=session_id,
                summary=summary,
                original_messages=messages,
                token_count=0,  # No longer tracking tokens
                importance_score=importance_score,
                metadata={
                    "compressed_at": int(time.time() * 1000),
                    "message_count": message_count
                }
            )
            
            logger.info(f"Compressed {message_count} messages to long-term (ID: {summary_id}, score: {importance_score:.2f})")
            
            key = self.short_memory._get_session_key()
            self.short_memory.storage.trim(key, message_count, -1)
            logger.info(f"Trimmed {message_count} compressed messages from short-term memory")
            
        except Exception as e:
            logger.error(f"Failed to compress memory: {e}")
    
    async def process_user_message(
        self,
        user_message: str,
        user_emotion: str = "normal",
        user_lang: str = "en",
        source: str = "mic",
        interrupt: bool = False
    ) -> List[Dict[str, Any]]:
        """Process user input: retrieve memories → build prompt → generate → save."""
        async with self._lock:
            self.is_processing = True
            self.last_interaction_time = time.time()
            
            try:
                long_term_summaries = await self._retrieve_long_term_summaries()
                
                self.short_memory.add_user_message(
                    message=user_message,
                    emotion=user_emotion,
                    lang=user_lang,
                    source=source,
                    interrupt=interrupt
                )
                logger.info(f"User message saved: {user_message[:50]}...")
                
                conversation_history = self.short_memory.get_conversation_context()
                
                messages = self.prompt_builder.build_prompt(
                    user_message=user_message,
                    conversation_history=conversation_history,
                    is_auto_trigger=False
                )
                
                insert_position = 1
                if long_term_summaries:
                    messages.insert(insert_position, {
                        "role": "system",
                        "content": long_term_summaries
                    })
                    insert_position += 1
                
                start_time = time.time()
                response_sentences = await asyncio.to_thread(
                    self.llm.generate_and_parse, messages
                )
                latency_ms = int((time.time() - start_time) * 1000)
                
                logger.info(f"LLM response generated: {len(response_sentences)} sentences in {latency_ms}ms")
                
                saved_responses = []
                for idx, sentence in enumerate(response_sentences):
                    response_entry = self.short_memory.add_chino_response(
                        text_spoken=sentence.get("text_spoken", ""),
                        text_display=sentence.get("text_display", ""),
                        lang="jp",
                        emotion=sentence.get("emo", "normal"),
                        action=sentence.get("act", "none"),
                        intensity=sentence.get("intensity", 0.5),
                        stream_index=idx,
                        is_completed=(idx == len(response_sentences) - 1),
                        latency_ms=latency_ms if idx == 0 else 0
                    )
                    saved_responses.append(response_entry)
                
                asyncio.create_task(self._check_and_compress_memory())
                
                self.log_memory_stats()
                
                return saved_responses
                
            except Exception as e:
                logger.error(f"Error processing user message: {e}")
                raise
            finally:
                self.is_processing = False
    
    async def auto_trigger_conversation(self) -> List[Dict[str, Any]]:
        """Auto-trigger conversation when idle timeout reached."""
        if self._lock.locked():
            logger.debug("Skipping auto-trigger: already processing")
            return []
        
        async with self._lock:
            self.is_processing = True
            
            try:
                logger.info("Auto-trigger: Initiating conversation")
                
                long_term_summaries = await self._retrieve_long_term_summaries()
                
                conversation_history = self.short_memory.get_conversation_context()
                
                messages = self.prompt_builder.build_prompt(
                    user_message=None,
                    conversation_history=conversation_history,
                    is_auto_trigger=True
                )
                
                if long_term_summaries:
                    messages.insert(1, {
                        "role": "system",
                        "content": long_term_summaries
                    })
                
                start_time = time.time()
                response_sentences = await asyncio.to_thread(
                    self.llm.generate_and_parse, messages
                )
                latency_ms = int((time.time() - start_time) * 1000)
                
                logger.info(f"Auto-trigger response: {len(response_sentences)} sentences in {latency_ms}ms")
                
                saved_responses = []
                for idx, sentence in enumerate(response_sentences):
                    response_entry = self.short_memory.add_chino_response(
                        text_spoken=sentence.get("text_spoken", ""),
                        text_display=sentence.get("text_display", ""),
                        lang="jp",
                        emotion=sentence.get("emo", "normal"),
                        action=sentence.get("act", "none"),
                        intensity=sentence.get("intensity", 0.5),
                        stream_index=idx,
                        is_completed=(idx == len(response_sentences) - 1),
                        latency_ms=latency_ms if idx == 0 else 0
                    )
                    saved_responses.append(response_entry)
                
                self.last_interaction_time = time.time()
                return saved_responses
                
            except Exception as e:
                logger.error(f"Error in auto-trigger: {e}")
                return []
            finally:
                self.is_processing = False
    
    async def start_auto_trigger_loop(self):
        """Start background task for auto-trigger monitoring."""
        logger.info(f"Starting auto-trigger loop (timeout: {self.idle_timeout}s)")
        
        while True:
            try:
                await asyncio.sleep(5)
                
                if self.is_processing:
                    continue
                
                idle_duration = time.time() - self.last_interaction_time
                
                if idle_duration >= self.idle_timeout:
                    logger.debug(f"Idle for {idle_duration:.1f}s, triggering auto-conversation")
                    await self.auto_trigger_conversation()
                
            except asyncio.CancelledError:
                logger.info("Auto-trigger loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in auto-trigger loop: {e}")
                await asyncio.sleep(5)
    
    def start_auto_trigger(self):
        """Start auto-trigger in background."""
        if self.auto_trigger_task is None or self.auto_trigger_task.done():
            self.auto_trigger_task = asyncio.create_task(self.start_auto_trigger_loop())
            logger.info("Auto-trigger task started")
    
    def stop_auto_trigger(self):
        """Stop auto-trigger background task."""
        if self.auto_trigger_task and not self.auto_trigger_task.done():
            self.auto_trigger_task.cancel()
            logger.info("Auto-trigger task stopped")
    
    @property
    def memory(self) -> ShortTermMemory:
        """Get short-term memory instance (for API compatibility)."""
        return self.short_memory
    
    def get_conversation_history(self, count: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieve recent conversation from memory."""
        return self.short_memory.get_recent_messages(count)
    
    def clear_conversation(self):
        """Clear all conversation history."""
        self.short_memory.clear()
        self.last_interaction_time = time.time()
        logger.info("Conversation history cleared")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about all memory layers."""
        stats = {
            "short_term": {
                "messages": len(self.short_memory.buffer)
            }
        }
        
        if self.long_memory:
            try:
                stats["long_term"] = {
                    "recent_summaries": len(self.long_memory.get_recent_summaries(limit=5)),
                    "total_summaries": self.long_memory.get_summary_count()
                }
            except Exception:
                stats["long_term"] = {"status": "error"}
        
        return stats
    
    def log_memory_stats(self, event_type: str = "message_processed"):
        """Log detailed memory statistics including total tokens.
        
        Args:
            event_type: Type of event triggering the log
        """
        stats = self.get_memory_stats()
        short_term = stats.get("short_term", {})
        
        logger.info("\n" + "="*60)
        logger.info("MEMORY STATISTICS")
        logger.info("="*60)
        logger.info(f"Short-term: {short_term.get('messages', 0)} messages, {short_term.get('tokens', 0)} tokens")
        
        if "long_term" in stats and "status" not in stats["long_term"]:
            long_term = stats["long_term"]
            logger.info(f"Long-term: {long_term.get('total_summaries', 0)} summaries (recent: {long_term.get('recent_summaries', 0)})")
        
        logger.info("="*60 + "\n")