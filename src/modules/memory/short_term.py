"""
Short-term memory: Conversation buffer storage with multiple backend support.
Stores recent messages following the expected JSON format.
Supports Redis and in-memory cache backends.
"""
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import uuid
import json
import logging

from src.modules.memory.cache.memory_cache import MemoryCache
from src.modules.memory.cache.redis_storage import RedisMemoryStorage

logger = logging.getLogger(__name__)


class ShortTermMemory:
    """Manages recent conversation history with configurable storage backend."""
    
    def __init__(
        self,
        max_size: int = 20,
        storage_type: str = "redis",
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        key_prefix: str = "chatbot:conversation"
    ):
        self.max_size = max_size
        self.storage_key = f"{key_prefix}:main"
        self.storage: Union[MemoryCache, RedisMemoryStorage]
        
        if storage_type == "in-memory":
            self.storage = MemoryCache()
            logger.info("Using MemoryCache backend")
        elif storage_type == "redis":
            self.storage = RedisMemoryStorage(
                host=redis_host,
                port=redis_port,
                db=redis_db
            )
            logger.info("Using RedisMemoryStorage backend")
        else:
            raise ValueError(
                f"Invalid storage_type: {storage_type}. Must be 'redis' or 'in-memory'"
            )
    
    def add_user_message(
        self, 
        message: str, 
        emotion: str = "normal",
        lang: str = "en",
        source: str = "mic",
        interrupt: bool = False
    ) -> Dict[str, Any]:
        """Add user message to conversation buffer."""
        entry = {
            "user": {
                "message": message,
                "emotion": emotion,
                "lang": lang,
                "context": {
                    "source": source,
                    "interrupt": interrupt
                },
                "meta": {
                    "timestamp": int(datetime.now().timestamp() * 1000)
                }
            }
        }
        
        message_json = json.dumps(entry, ensure_ascii=False)
        self.storage.add_message(self.storage_key, message_json)
        self.storage.trim(self.storage_key, -self.max_size, -1)
        
        return entry
    
    def add_chino_response(
        self,
        text_spoken: str,
        text_display: str,
        lang: str = "jp",
        emotion: str = "normal",
        action: str = "none",
        intensity: float = 0.5,
        stream_index: int = 0,
        is_completed: bool = True,
        latency_ms: int = 0
    ) -> Dict[str, Any]:
        """Add assistant response to conversation buffer."""
        entry = {
            "chino-kafuu": {
                "response_id": str(uuid.uuid4()),
                "stream_index": str(stream_index),
                "is_completed": is_completed,
                "message": {
                    "text_spoken": text_spoken,
                    "text_display": text_display,
                    "phonemes": ""
                },
                "tts": {
                    "lang": lang,
                    "emotion": emotion,
                    "action": action,
                    "intensity": intensity
                },
                "meta": {
                    "latency_ms": latency_ms,
                    "timestamp": int(datetime.now().timestamp() * 1000)
                }
            }
        }
        
        message_json = json.dumps(entry, ensure_ascii=False)
        self.storage.add_message(self.storage_key, message_json)
        self.storage.trim(self.storage_key, -self.max_size, -1)
        
        return entry
    
    def get_recent_messages(self, count: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieve recent messages from storage buffer."""
        if count is None:
            messages = self.storage.get_messages(self.storage_key, 0, -1)
        else:
            messages = self.storage.get_messages(self.storage_key, -count, -1)
        
        return [json.loads(msg) for msg in messages]
    
    def get_conversation_context(self) -> List[Dict[str, str]]:
        """Extract simplified conversation for LLM context."""
        messages = self.get_recent_messages()
        context = []
        
        for entry in messages:
            if "user" in entry:
                context.append({
                    "role": "user",
                    "content": entry["user"]["message"],
                    "emotion": entry["user"]["emotion"]
                })
            elif "chino-kafuu" in entry:
                context.append({
                    "role": "assistant", 
                    "content": entry["chino-kafuu"]["message"]["text_display"],
                    "emotion": entry["chino-kafuu"]["tts"]["emotion"]
                })
        
        return context
    
    def clear(self):
        """Clear all messages from buffer."""
        self.storage.delete(self.storage_key)
        logger.info("Conversation buffer cleared")
    
    @property
    def buffer(self):
        """Get buffer as list for compatibility."""
        return self.get_recent_messages()
    
    def __del__(self):
        """Cleanup storage connection."""
        if hasattr(self, 'storage') and hasattr(self.storage, 'close'):
            try:
                self.storage.close()
            except:
                pass