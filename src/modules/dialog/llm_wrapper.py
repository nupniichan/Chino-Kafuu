"""
LLM wrapper: Handles GGUF model inference using llama-cpp-python.
Provides streaming and non-streaming text gene  ration.
"""
import logging
from typing import List, Dict, Any, Iterator, Optional
import json

logger = logging.getLogger(__name__)

class LLMWrapper:
    """Wrapper for GGUF model inference via llama-cpp-python."""
    
    def __init__(
        self,
        model_path: str,
        n_ctx: int = 8192,
        n_gpu_layers: int = -1,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 512
    ):
        """Initialize LLM with model path and generation parameters."""
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.llm = None
        
        self._load_model()
    
    def _load_model(self):
        """Load GGUF model using llama-cpp-python."""
        try:
            from llama_cpp import Llama
            
            logger.info(f"Loading GGUF model from: {self.model_path}")
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_gpu_layers=self.n_gpu_layers,
                verbose=False
            )
            logger.info("GGUF model loaded successfully")
            
        except ImportError:
            logger.error("llama-cpp-python not installed")
            raise
        except Exception as e:
            logger.error(f"Failed to load GGUF model: {e}")
            raise
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate complete response from messages."""
        if not self.llm:
            raise RuntimeError("Model not loaded")
        
        try:
            response = self.llm.create_chat_completion(
                messages=messages,
                temperature=temperature or self.temperature,
                top_p=self.top_p,
                max_tokens=max_tokens or self.max_tokens,
                stream=False
            )
            
            return response["choices"][0]["message"]["content"]
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
    
    def generate_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Iterator[str]:
        """Generate streaming response from messages."""
        if not self.llm:
            raise RuntimeError("Model not loaded")
        
        try:
            stream = self.llm.create_chat_completion(
                messages=messages,
                temperature=temperature or self.temperature,
                top_p=self.top_p,
                max_tokens=max_tokens or self.max_tokens,
                stream=True
            )
            
            for chunk in stream:
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    yield content
                    
        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            raise
    
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
    
    def __del__(self):
        """Cleanup model on deletion."""
        if self.llm:
            del self.llm
            logger.info("LLM model unloaded")