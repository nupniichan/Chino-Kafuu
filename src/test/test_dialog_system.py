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

from modules.dialog.llm_wrapper import LLMWrapper
from modules.dialog.orchestrator import DialogOrchestrator
from modules.memory.short_term import ShortTermMemory
from setting import (
    LLM_MODEL_PATH, 
    LLM_N_CTX, 
    LLM_N_GPU_LAYERS,
    LLM_TEMPERATURE,
    LLM_TOP_P,
    LLM_MAX_TOKENS,
    SHORT_TERM_MEMORY_SIZE,
    IDLE_TIMEOUT_SECONDS
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_dialog_flow():
    """Test complete dialog flow with memory and LLM."""
    
    try:
        logger.info("Initializing LLM wrapper...")
        llm = LLMWrapper(
            model_path=LLM_MODEL_PATH,
            n_ctx=LLM_N_CTX,
            n_gpu_layers=LLM_N_GPU_LAYERS,
            temperature=LLM_TEMPERATURE,
            top_p=LLM_TOP_P,
            max_tokens=LLM_MAX_TOKENS
        )
        
        logger.info("Initializing dialog orchestrator...")
        memory = ShortTermMemory(max_size=SHORT_TERM_MEMORY_SIZE)
        orchestrator = DialogOrchestrator(
            llm_wrapper=llm,
            short_term_memory=memory,
            idle_timeout=IDLE_TIMEOUT_SECONDS
        )
        
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
            logger.info("--- Start of Redis Memory ---")
            for i, entry in enumerate(history):
                # Pretty print the JSON content
                logger.info(f"[{i+1}]:\n{json.dumps(entry, indent=2, ensure_ascii=False)}")
            logger.info("--- End of Redis Memory ---")
        
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
        
        llm = LLMWrapper(
            model_path=LLM_MODEL_PATH,
            n_ctx=8192,
            temperature=0.7,
            max_tokens=256
        )
        
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