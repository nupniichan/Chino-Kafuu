"""
Short-term memory: Redis-based conversation buffer storage.
Stores recent messages following the expected JSON format.
"""
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid
import json
import logging

logger = logging.getLogger(__name__)


class ShortTermMemory:
    """Manages recent conversation history in Redis."""
    
    def __init__(
        self,
        max_size: int = 20,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        key_prefix: str = "chatbot:conversation"
    ):
        """Initialize with Redis connection and buffer settings."""
        self.max_size = max_size
        self.key_prefix = key_prefix
        self.current_session_id: Optional[str] = None
        self.redis_client = None
        
        self._connect_redis(redis_host, redis_port, redis_db)
    
    def _connect_redis(self, host: str, port: int, db: int):
        """Connect to Redis server."""
        try:
            import redis
            
            self.redis_client = redis.Redis(
                host=host,
                port=port,
                db=db,
                decode_responses=True
            )
            
            self.redis_client.ping()
            logger.info(f"Connected to Redis at {host}:{port}")
            
        except ImportError:
            logger.error("redis package not installed. Install: pip install redis")
            raise
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    def _get_session_key(self) -> str:
        """Get Redis key for current session."""
        session_id = self.current_session_id or "default"
        return f"{self.key_prefix}:{session_id}"
    
    def add_user_message(
        self, 
        message: str, 
        emotion: str = "normal",
        lang: str = "en",
        source: str = "mic",
        interrupt: bool = False
    ) -> Dict[str, Any]:
        """Add user message to conversation buffer."""
        if not self.current_session_id:
            self.start_new_session()
        
        entry = {
            "user": {
                "message": message,
                "emotion": emotion,
                "lang": lang,
                "context": {
                    "session_id": self.current_session_id,
                    "source": source,
                    "interrupt": interrupt
                },
                "meta": {
                    "timestamp": int(datetime.now().timestamp() * 1000)
                }
            }
        }
        
        key = self._get_session_key()
        self.redis_client.rpush(key, json.dumps(entry, ensure_ascii=False))
        self.redis_client.ltrim(key, -self.max_size, -1)
        
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
        if not self.current_session_id:
            self.start_new_session()
        
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
        
        key = self._get_session_key()
        self.redis_client.rpush(key, json.dumps(entry, ensure_ascii=False))
        self.redis_client.ltrim(key, -self.max_size, -1)
        
        return entry
    
    def get_recent_messages(self, count: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieve recent messages from Redis buffer."""
        key = self._get_session_key()
        
        if count is None:
            messages = self.redis_client.lrange(key, 0, -1)
        else:
            messages = self.redis_client.lrange(key, -count, -1)
        
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
                    "role": "Chino", 
                    "content": entry["chino-kafuu"]["message"]["text_display"],
                    "emotion": entry["chino-kafuu"]["tts"]["emotion"]
                })
        
        return context
    
    def clear(self):
        """Clear all messages from buffer."""
        if self.current_session_id:
            key = self._get_session_key()
            self.redis_client.delete(key)
            logger.info(f"Cleared session: {self.current_session_id}")
        self.current_session_id = None
    
    def start_new_session(self) -> str:
        """Start a new conversation session."""
        self.current_session_id = self._generate_session_id()
        logger.info(f"Started new session: {self.current_session_id}")
        return self.current_session_id
    
    @staticmethod
    def _generate_session_id() -> str:
        """Generate unique session ID with timestamp."""
        return f"session_{int(datetime.now().timestamp())}"
    
    @property
    def buffer(self):
        """Get buffer as list for compatibility."""
        return self.get_recent_messages()
    
    def __del__(self):
        """Cleanup Redis connection."""
        if self.redis_client:
            try:
                self.redis_client.close()
            except:
                pass