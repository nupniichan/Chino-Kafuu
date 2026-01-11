"""
Redis storage: Redis-based memory storage implementation.
Provides persistent storage using Redis server.
"""
from typing import List
import logging
from modules.memory.storage import MemoryStorage

logger = logging.getLogger(__name__)


class RedisMemoryStorage(MemoryStorage):
    """Redis-based memory storage implementation."""
    
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        """Initialize Redis connection."""
        self.host = host
        self.port = port
        self.db = db
        self.redis_client = None
        self._connect_redis()
    
    def _connect_redis(self):
        """Connect to Redis server."""
        try:
            import redis
            
            self.redis_client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                decode_responses=True
            )
            
            self.redis_client.ping()
            logger.info(f"Connected to Redis at {self.host}:{self.port}")
            
        except ImportError:
            logger.error("redis package not installed. Install: pip install redis")
            raise
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    def add_message(self, key: str, message: str) -> None:
        """Add a message to Redis list."""
        self.redis_client.rpush(key, message)
    
    def get_messages(self, key: str, start: int = 0, end: int = -1) -> List[str]:
        """Retrieve messages from Redis list."""
        return self.redis_client.lrange(key, start, end)
    
    def trim(self, key: str, start: int, end: int) -> None:
        """Trim Redis list to specified range."""
        self.redis_client.ltrim(key, start, end)
    
    def delete(self, key: str) -> None:
        """Delete a key from Redis."""
        self.redis_client.delete(key)
    
    def exists(self, key: str) -> bool:
        """Check if key exists in Redis."""
        return bool(self.redis_client.exists(key))
    
    def close(self):
        """Close Redis connection."""
        if self.redis_client:
            try:
                self.redis_client.close()
            except:
                pass
    
    def __del__(self):
        """Cleanup Redis connection."""
        self.close()
