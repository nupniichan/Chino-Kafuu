"""
Test memory compression with low token limit.
Verifies that memory compresses to long-term when token limit is reached.
"""
import sys
import os
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.dialog.orchestrator import DialogOrchestrator
from modules.dialog.llm_wrapper import OpenRouterLLMWrapper
from modules.memory.short_term import ShortTermMemory
from modules.memory.long_term import LongTermMemory
from setting import (
    OPENROUTER_API_KEY,
    OPENROUTER_MODEL,
    OPENROUTER_BASE_URL,
    SHORT_TERM_TOKEN_LIMIT
)


async def test_memory_compression():
    """Test memory compression with multiple messages."""
    
    print("\n" + "="*70)
    print("🧪 TESTING MEMORY COMPRESSION")
    print("="*70)
    print(f"Token limit: {SHORT_TERM_TOKEN_LIMIT}")
    print("-"*70 + "\n")
    
    # Initialize components
    llm = OpenRouterLLMWrapper(
        api_key=OPENROUTER_API_KEY,
        model=OPENROUTER_MODEL,
        base_url=OPENROUTER_BASE_URL
    )
    
    short_memory = ShortTermMemory(max_size=20, storage_type="in-memory")
    long_memory = LongTermMemory()
    
    orchestrator = DialogOrchestrator(
        llm_wrapper=llm,
        short_term_memory=short_memory,
        long_term_memory=long_memory,
        token_limit=SHORT_TERM_TOKEN_LIMIT,
        importance_threshold=0.8
    )
    
    # Send multiple messages to exceed token limit
    test_messages = [
        "Hello Chino! How are you today?",
        "What's your favorite drink to serve at Rabbit House?",
        "Tell me about your friends Rize and Cocoa.",
        "What do you like to do in your free time?",
        "Can you recommend a good coffee blend?",
    ]
    
    print("📨 Sending test messages...\n")
    
    for i, msg in enumerate(test_messages, 1):
        print(f"[{i}] User: {msg}")
        
        try:
            responses = await orchestrator.process_user_message(
                user_message=msg,
                user_emotion="normal",
                user_lang="en"
            )
            
            if responses:
                print(f"    Chino: {responses[0].get('chino-kafuu', {}).get('message', {}).get('text_display', 'No response')[:60]}...")
            
            # Check memory stats
            stats = orchestrator.get_memory_stats()
            short_term = stats.get("short_term", {})
            tokens = short_term.get("tokens", 0)
            messages = short_term.get("messages", 0)
            
            print(f"    📊 Memory: {messages} messages, {tokens} tokens")
            
            if tokens >= SHORT_TERM_TOKEN_LIMIT:
                print(f"    ⚠️ Token limit reached! Should compress...")
            
            print()
            
            # Wait a bit between messages
            await asyncio.sleep(1)
            
        except Exception as e:
            print(f"    ❌ Error: {e}\n")
    
    # Final stats
    print("\n" + "="*70)
    print("📊 FINAL MEMORY STATISTICS")
    print("="*70)
    
    orchestrator.log_memory_stats()
    
    # Check long-term memory
    long_summaries = long_memory.get_recent_summaries(limit=10)
    print(f"\n📚 Long-term summaries created: {len(long_summaries)}")
    
    if long_summaries:
        print("\nSummaries:")
        for summary in long_summaries:
            print(f"  - ID {summary['id']}: {summary['summary'][:80]}...")
            print(f"    Importance: {summary['importance_score']:.2f}, Tokens: {summary['token_count']}")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    asyncio.run(test_memory_compression())
