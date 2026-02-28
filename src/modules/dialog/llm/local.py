"""
Local LLM wrapper: Handles GGUF model inference using llama-cpp-python.
"""
import logging
from typing import List, Dict, Iterator, Optional

from src.modules.dialog.llm_wrapper import BaseLLMWrapper

logger = logging.getLogger(__name__)


class LocalLLMWrapper(BaseLLMWrapper):
    """Wrapper for GGUF model inference via llama-cpp-python."""
    
    def __init__(
        self,
        model_path: str,
        n_ctx: int = 8192,
        n_gpu_layers: int = 20,
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
    
    def __del__(self):
        """Cleanup model on deletion."""
        if self.llm:
            del self.llm
            logger.info("LLM model unloaded")