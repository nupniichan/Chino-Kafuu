"""
OpenRouter LLM wrapper: Handles remote LLM inference via OpenRouter API.
"""
import logging
import json
from typing import List, Dict, Iterator, Optional

from src.modules.dialog.llm_wrapper import BaseLLMWrapper

logger = logging.getLogger(__name__)


class OpenRouterLLMWrapper(BaseLLMWrapper):
    """Wrapper for OpenRouter API remote LLM inference."""
    
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://openrouter.ai/api/v1",
        timeout: int = 60,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 512
    ):
        """Initialize OpenRouter LLM with API credentials and parameters."""
        if not api_key:
            raise ValueError("OpenRouter API key is required")
        
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        
        try:
            import requests
            self.requests = requests
        except ImportError:
            logger.error("requests library not installed. Install with: pip install requests")
            raise
        
        logger.info(f"Initialized OpenRouter LLM with model: {self.model}")
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate complete response from messages via OpenRouter API."""
        url = f"{self.base_url}/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/ChinoKafuu",
            "X-Title": "ChinoKafuu AI"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "top_p": self.top_p,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens
        }
        
        try:
            response = self.requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
        except self.requests.exceptions.RequestException as e:
            logger.error(f"OpenRouter API request failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_detail = e.response.json()
                    logger.error(f"Error response: {error_detail}")
                except:
                    logger.error(f"Error response text: {e.response.text}")
            raise
        except KeyError as e:
            logger.error(f"Unexpected response format from OpenRouter: {e}")
            raise
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
    
    def generate_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Iterator[str]:
        """Generate streaming response from messages via OpenRouter API."""
        url = f"{self.base_url}/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/ChinoKafuu",
            "X-Title": "ChinoKafuu AI"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "top_p": self.top_p,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
            "stream": True
        }
        
        try:
            response = self.requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=self.timeout,
                stream=True
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if not line:
                    continue
                
                line_str = line.decode('utf-8')
                if line_str.startswith('data: '):
                    data_str = line_str[6:]
                    if data_str == '[DONE]':
                        break
                    
                    try:
                        chunk_data = json.loads(data_str)
                        delta = chunk_data.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        continue
                        
        except self.requests.exceptions.RequestException as e:
            logger.error(f"OpenRouter API streaming request failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_detail = e.response.json()
                    logger.error(f"Error response: {error_detail}")
                except:
                    logger.error(f"Error response text: {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            raise