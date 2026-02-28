"""
Memory cache: In-memory Python dict-based storage implementation.
Provides fast, simple storage without external dependencies.
"""
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class MemoryCache:
    """In-memory cache storage using Python dict."""
    
    def __init__(self):
        """Initialize in-memory storage."""
        self._storage: Dict[str, List[str]] = {}
        logger.info("MemoryCache initialized")
    
    def add_message(self, key: str, message: str) -> None:
        """Add a message to the cache."""
        if key not in self._storage:
            self._storage[key] = []
        self._storage[key].append(message)
    
    def _resolve_range(self, length: int, start: int, end: int) -> tuple:
        """Resolve Redis-style inclusive range to Python slice indices."""
        if start < 0:
            start = max(length + start, 0)
        if end < 0:
            end = length + end
        return start, end + 1

    def get_messages(self, key: str, start: int = 0, end: int = -1) -> List[str]:
        """Retrieve messages from cache (Redis LRANGE compatible)."""
        if key not in self._storage:
            return []
        
        messages = self._storage[key]
        s, e = self._resolve_range(len(messages), start, end)
        return messages[s:e]
    
    def trim(self, key: str, start: int, end: int) -> None:
        """Trim messages list to specified range."""
        if key not in self._storage:
            return
        
        messages = self._storage[key]
        s, e = self._resolve_range(len(messages), start, end)
        self._storage[key] = messages[s:e]
    
    def delete(self, key: str) -> None:
        """Delete a key from cache."""
        if key in self._storage:
            del self._storage[key]
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        return key in self._storage
    
    def clear_all(self):
        """Clear all cached data."""
        self._storage.clear()
        logger.info("MemoryCache cleared")
