"""
Prompt builder: Constructs system and user prompts with memory context.
Builds prompts based on conversation history and character rules.
"""
from typing import List, Dict, Any, Optional
import os


class PromptBuilder:
    """Builds prompts for LLM with conversation context and character rules."""
    
    def __init__(self, prompt_file: Optional[str] = None):
        """Initialize with optional custom prompt file path."""
        if prompt_file is None:
            prompt_file = os.path.join(
                os.path.dirname(__file__), 
                "prompt.txt"
            )
        self.prompt_file = prompt_file
        self.prompt = self._load_prompt()
    
    def _load_prompt(self) -> str:
        """Load prompt from file."""
        try:
            with open(self.prompt_file, "r", encoding="utf-8") as f:
                return f.read().strip()
        except FileNotFoundError:
            raise FileNotFoundError(f"Prompt file not found: {self.prompt_file}")
        except Exception as e:
            raise RuntimeError(f"Failed to load prompt: {e}")
    
    def build_prompt(
        self, 
        user_message: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        is_auto_trigger: bool = False
    ) -> List[Dict[str, str]]:
        """Build complete prompt with system, history, and user message."""
        messages = [{"role": "system", "content": self.prompt}]
        
        if conversation_history:
            history_limit = 20
            recent_history = conversation_history[-history_limit:]
            
            for msg in recent_history:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                emotion = msg.get("emotion", "normal")
                
                if role == "user":
                    messages.append({
                        "role": "user",
                        "content": f"[User emotion: {emotion}] {content}"
                    })
                else:
                    messages.append({
                        "role": "Chino",
                        "content": content
                    })
        
        if is_auto_trigger and not user_message:
            messages.append({
                "role": "user",
                "content": "[Auto-trigger: User has been quiet. Initiate conversation naturally.]"
            })
        elif user_message:
            messages.append({
                "role": "user",
                "content": user_message
            })
        
        return messages
    
    def format_conversation_summary(self, history: List[Dict[str, str]]) -> str:
        """Format conversation history into readable summary."""
        if not history:
            return "No previous conversation."
        
        summary_lines = []
        for msg in history[-5:]:
            role = "User" if msg.get("role") == "user" else "Chino"
            content = msg.get("content", "")
            summary_lines.append(f"{role}: {content}")
        
        return "\n".join(summary_lines)