"""
Memory storage abstractions: Abstract interface for storage backends.
Provides strategy pattern for different memory storage implementations.
"""
from abc import ABC, abstractmethod
from typing import List

logger = None


def _get_logger():
    """Lazy import logger to avoid circular imports."""
    global logger
    if logger is None:
        import logging
        logger = logging.getLogger(__name__)
    return logger


class MemoryStorage(ABC):
    """Abstract base class for memory storage backends."""
    
    @abstractmethod
    def add_message(self, key: str, message: str) -> None:
        """Add a message to the storage."""
        pass
    
    @abstractmethod
    def get_messages(self, key: str, start: int = 0, end: int = -1) -> List[str]:
        """Retrieve messages from storage."""
        pass
    
    @abstractmethod
    def trim(self, key: str, start: int, end: int) -> None:
        """Trim messages list to specified range."""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete a key from storage."""
        pass
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists in storage."""
        pass

