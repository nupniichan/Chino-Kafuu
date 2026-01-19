"""
Test dialog system: Test LLM integration with memory and auto-trigger.
Run this to test the complete dialog flow.
"""
import asyncio
import logging
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.dialog.llm_wrapper import LocalLLMWrapper, OpenRouterLLMWrapper
from modules.dialog.orchestrator import DialogOrchestrator
from modules.memory.short_term import ShortTermMemory
from modules.memory.long_term import LongTermMemory
from setting import (
    LLM_MODE,
    LLM_MODEL_PATH,
    LLM_N_CTX,
    LLM_N_GPU_LAYERS,
    LLM_TEMPERATURE,
    LLM_TOP_P,
    LLM_MAX_TOKENS,
    OPENROUTER_API_KEY,
    OPENROUTER_MODEL,
    OPENROUTER_BASE_URL,
    OPENROUTER_TIMEOUT,
    SHORT_TERM_MEMORY_SIZE,
    SHORT_TERM_TOKEN_LIMIT,
    MEMORY_IMPORTANCE_THRESHOLD,
    IDLE_TIMEOUT_SECONDS,
    MEMORY_CACHE,
    REDIS_HOST,
    REDIS_PORT,
    REDIS_DB
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_llm_instance():
    """Create LLM instance based on LLM_MODE configuration."""
    if LLM_MODE == "local":
        return LocalLLMWrapper(
            model_path=LLM_MODEL_PATH,
            n_ctx=LLM_N_CTX,
            n_gpu_layers=LLM_N_GPU_LAYERS,
            temperature=LLM_TEMPERATURE,
            top_p=LLM_TOP_P,
            max_tokens=LLM_MAX_TOKENS
        )
    elif LLM_MODE == "openrouter":
        return OpenRouterLLMWrapper(
            api_key=OPENROUTER_API_KEY,
            model=OPENROUTER_MODEL,
            base_url=OPENROUTER_BASE_URL,
            timeout=OPENROUTER_TIMEOUT,
            temperature=LLM_TEMPERATURE,
            top_p=LLM_TOP_P,
            max_tokens=LLM_MAX_TOKENS
        )
    else:
        raise ValueError(f"Invalid LLM_MODE: {LLM_MODE}. Must be 'local' or 'openrouter'")


async def test_dialog_flow():
    """Test complete dialog flow with memory and LLM."""
    
    try:
        logger.info("Initializing LLM wrapper...")
        llm = create_llm_instance()
        
        logger.info("Initializing dialog orchestrator...")
        
        short_memory = ShortTermMemory(
            max_size=SHORT_TERM_MEMORY_SIZE,
            storage_type=MEMORY_CACHE,
            redis_host=REDIS_HOST,
            redis_port=REDIS_PORT,
            redis_db=REDIS_DB
        )
        long_memory = LongTermMemory()
        
        orchestrator = DialogOrchestrator(
            llm_wrapper=llm,
            short_term_memory=short_memory,
            long_term_memory=long_memory,
            idle_timeout=IDLE_TIMEOUT_SECONDS,
            token_limit=SHORT_TERM_TOKEN_LIMIT,
            importance_threshold=MEMORY_IMPORTANCE_THRESHOLD
        )
        logger.info(f"Token limit: {SHORT_TERM_TOKEN_LIMIT}, Importance threshold: {MEMORY_IMPORTANCE_THRESHOLD}")
        
        logger.info("\n" + "="*60)
        logger.info("Testing User Message Processing")
        logger.info("="*60)
        
        test_message = "Chino~, how's your day?"
        logger.info(f"\nUser: {test_message}")
        
        responses = await orchestrator.process_user_message(
            user_message=test_message,
            user_emotion="happy",
            user_lang="vi",
            source="text"
        )
        
        logger.info(f"\nChino responded with {len(responses)} sentences:")
        for idx, resp in enumerate(responses):
            chino = resp.get("chino-kafuu", {})
            message = chino.get("message", {})
            tts = chino.get("tts", {})
            
            logger.info(f"\n[{idx+1}] JP: {message.get('text_spoken', '')}")
            logger.info(f"    VI: {message.get('text_display', '')}")
            logger.info(f"    Emotion: {tts.get('emotion', '')}, Action: {tts.get('act', '')}")
        
        logger.info("\n" + "="*60)
        logger.info("Conversation History")
        logger.info("="*60)
        
        history = orchestrator.get_conversation_history()
        logger.info(f"Total messages in memory: {len(history)}")
        if history:
            logger.info("--- Start of Memory ---")
            for i, entry in enumerate(history):
                # Pretty print the JSON content
                logger.info(f"[{i+1}]:\n{json.dumps(entry, indent=2, ensure_ascii=False)}")
            logger.info("--- End of Memory ---")
        
        logger.info("\n" + "="*60)
        logger.info("Testing Auto-Trigger (waiting for idle timeout...)")
        logger.info("="*60)
        
        orchestrator.start_auto_trigger()
        
        await asyncio.sleep(IDLE_TIMEOUT_SECONDS + 5)
        
        history_after_trigger = orchestrator.get_conversation_history()
        logger.info(f"\nMessages after auto-trigger: {len(history_after_trigger)}")
        
        if len(history_after_trigger) > len(history):
            logger.info("✓ Auto-trigger worked! Chino initiated conversation.")
            
            new_responses = history_after_trigger[len(history):]
            for resp in new_responses:
                if "chino-kafuu" in resp:
                    chino = resp.get("chino-kafuu", {})
                    message = chino.get("message", {})
                    logger.info(f"\nChino (auto): {message.get('text_display', '')}")
        else:
            logger.warning("Auto-trigger did not fire yet")
        
        orchestrator.stop_auto_trigger()
        logger.info("\n" + "="*60)
        logger.info("Test completed successfully!")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        raise


async def test_simple_generation():
    """Test simple LLM generation without orchestrator."""
    
    try:
        logger.info("Testing simple LLM generation...")
        
        llm = create_llm_instance()
        
        messages = [
            {
                "role": "system",
                "content": "You are Chino Kafuu. Respond in one sentence in Vietnamese."
            },
            {
                "role": "user",
                "content": "Xin chào!"
            }
        ]
        
        response = llm.generate(messages)
        logger.info(f"\nLLM Response: {response}")
        
    except Exception as e:
        logger.error(f"Simple test failed: {e}", exc_info=True)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test dialog system")
    parser.add_argument(
        "--mode",
        choices=["full", "simple"],
        default="full",
        help="Test mode: full (with orchestrator) or simple (LLM only)"
    )
    
    args = parser.parse_args()
    
    if args.mode == "simple":
        asyncio.run(test_simple_generation())
    else:
        asyncio.run(test_dialog_flow())