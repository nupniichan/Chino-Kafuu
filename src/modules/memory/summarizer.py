"""
Summarizer: Compresses conversation history using small LLM.
Used when short-term memory exceeds token limit.
"""
import logging
import os
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class ConversationSummarizer:
    """Summarizes conversations using lightweight LLM."""
    
    def __init__(self, llm_wrapper, prompt_file: Optional[str] = None):
        """Initialize with LLM wrapper instance."""
        self.llm = llm_wrapper
        
        if prompt_file is None:
            prompt_file = os.path.join(
                os.path.dirname(__file__),
                "summary_prompt.txt"
            )
        self.prompt_file = prompt_file
        self.system_prompt = self._load_prompt()
    
    def _load_prompt(self) -> str:
        """Load summarization prompt from file."""
        try:
            with open(self.prompt_file, "r", encoding="utf-8") as f:
                return f.read().strip()
        except FileNotFoundError:
            logger.error(f"Prompt file not found: {self.prompt_file}")
            raise
        except Exception as e:
            logger.error(f"Failed to load prompt: {e}")
            raise
    
    def summarize_conversation(
        self,
        messages: List[Dict[str, Any]],
        context: str = ""
    ) -> str:
        """Summarize conversation messages into concise text."""
        try:
            conversation_text = self._format_messages(messages)
            
            prompt_messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"Summarize this conversation:\n\n{conversation_text}"}
            ]
            
            if context:
                prompt_messages[1]["content"] += f"\n\nPrevious context: {context}"
            
            summary = self.llm.generate(
                messages=prompt_messages,
                temperature=0.3,
                max_tokens=200
            )
            
            logger.info(f"Generated summary: {summary[:100]}...")
            return summary.strip()
            
        except Exception as e:
            logger.error(f"Failed to summarize conversation: {e}")
            return self._fallback_summary(messages)
    
    def _format_messages(self, messages: List[Dict[str, Any]]) -> str:
        """Format messages into readable text."""
        formatted_lines = []
        
        for msg in messages:
            if "user" in msg:
                user_data = msg["user"]
                text = user_data.get("message", "")
                emotion = user_data.get("emotion", "normal")
                formatted_lines.append(f"User [{emotion}]: {text}")
                
            elif "chino-kafuu" in msg:
                chino_data = msg["chino-kafuu"]
                message_data = chino_data.get("message", {})
                tts_data = chino_data.get("tts", {})
                
                text = message_data.get("text_display", "")
                emotion = tts_data.get("emotion", "normal")
                formatted_lines.append(f"Assistant [{emotion}]: {text}")
        
        return "\n".join(formatted_lines)
    
    def _fallback_summary(self, messages: List[Dict[str, Any]]) -> str:
        """Create basic fallback summary if LLM fails."""
        user_messages = []
        chino_message = []
        
        for msg in messages:
            if "user" in msg:
                user_messages.append(msg["user"].get("message", ""))
            elif "chino-kafuu" in msg:
                chino_message.append(
                    msg["chino-kafuu"]["message"].get("text_display", "")
                )
        
        return f"Conversation with {len(user_messages)} user messages and {len(chino_message)} assistant responses."
    
    def calculate_importance_score(
        self,
        messages: List[Dict[str, Any]],
        summary: str
    ) -> float:
        """Calculate importance score for conversation (0.0 - 1.0)."""
        score = 0.5
        
        emotional_keywords = [
            "vui", "buồn", "giận", "hạnh phúc", "lo lắng",
            "happy", "sad", "angry", "worried", "excited"
        ]
        
        important_keywords = [
            "quan trọng", "nhớ", "yêu", "ghét", "cảm ơn",
            "important", "remember", "love", "hate", "thank"
        ]
        
        summary_lower = summary.lower()
        
        for keyword in emotional_keywords:
            if keyword in summary_lower:
                score += 0.1
        
        for keyword in important_keywords:
            if keyword in summary_lower:
                score += 0.15
        
        if len(messages) >= 10:
            score += 0.1
        
        return min(1.0, score)
