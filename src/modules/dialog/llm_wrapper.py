import logging
from typing import List, Dict, Any, Optional
import json

logger = logging.getLogger(__name__)


class BaseLLMWrapper:
    """Base class for LLM wrappers with shared parsing logic."""
    
    def parse_ndjson_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse NDJSON response into list of sentence objects."""
        sentences = []
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            try:
                if line.startswith('```') or line.endswith('```'):
                    line = line.strip('`').strip()
                    if not line or line == 'json':
                        continue
                
                sentence = json.loads(line)
                
                required_keys = ["user_emo", "text_spoken", "text_display", "emo", "act", "intensity"]
                if all(key in sentence for key in required_keys):
                    sentences.append(sentence)
                else:
                    logger.warning(f"Incomplete sentence structure: {line}")
                    
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON line: {line} - {e}")
                continue
        
        return sentences
    
    def generate_and_parse(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Generate response and parse NDJSON format."""
        response = self.generate(messages, temperature, max_tokens)
        return self.parse_ndjson_response(response)


from src.modules.dialog.llm.local import LocalLLMWrapper
from src.modules.dialog.llm.openrouter import OpenRouterLLMWrapper

LLMWrapper = LocalLLMWrapper

__all__ = [
    "BaseLLMWrapper",
    "LocalLLMWrapper",
    "OpenRouterLLMWrapper",
    "LLMWrapper"
]
