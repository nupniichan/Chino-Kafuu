"""
Token counter: Estimate token count for memory management.
Uses tiktoken for accurate token counting.
"""
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class TokenCounter:
    """Counts tokens in text for memory management."""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        """Initialize with encoding model."""
        self.encoding = None
        self._load_encoding(model_name)
    
    def _load_encoding(self, model_name: str):
        """Load tiktoken encoding."""
        try:
            import tiktoken
            self.encoding = tiktoken.encoding_for_model(model_name)
            logger.info(f"Token counter initialized with {model_name} encoding")
        except ImportError:
            logger.warning("tiktoken not installed, using fallback estimation")
            self.encoding = None
        except Exception as e:
            logger.warning(f"Failed to load tiktoken: {e}, using fallback")
            self.encoding = None
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self.encoding:
            return len(self.encoding.encode(text))
        else:
            return self._estimate_tokens(text)
    
    def count_messages_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """Count tokens in conversation messages."""
        total = 0
        for msg in messages:
            if "user" in msg:
                total += self.count_tokens(msg["user"].get("message", ""))
            elif "chino-kafuu" in msg:
                total += self.count_tokens(msg["chino-kafuu"]["message"].get("text_display", ""))
        return total
    
    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Fallback token estimation (rough approximation)."""
        return len(text) // 4