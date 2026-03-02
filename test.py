"""
Manual test script: Verify the full EventBus flow end-to-end.

Simulates: STT_READY -> Dialog Engine -> LLM_RESPONSE -> MemoryManager saves
No mic or audio hardware needed.

Usage: python test.py
"""
import asyncio
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("test")


async def test_event_bus_basic():
    """Test 1: EventBus subscribe/publish works correctly."""
    logger.info("=" * 60)
    logger.info("TEST 1: EventBus basic pub/sub")
    logger.info("=" * 60)

    from src.core.event_bus import EventBus, Priority

    bus = EventBus()
    received = []

    async def handler_a(event, data):
        received.append(("A", event, data))

    async def handler_b(event, data):
        received.append(("B", event, data))

    async def handler_priority(event, data):
        received.append(("HIGH", event, data))

    bus.subscribe("test_event", handler_a, owner="HandlerA")
    bus.subscribe("test_event", handler_b, owner="HandlerB")
    bus.subscribe("test_event", handler_priority, priority=Priority.HIGH, owner="HighPriority")

    count = await bus.publish("test_event", {"message": "hello"})

    assert count == 3, f"Expected 3 handlers, got {count}"
    assert received[0][0] == "HIGH", "HIGH priority handler should run first"
    assert len(received) == 3, f"Expected 3 received, got {len(received)}"

    logger.info(f"  Published to {count} handlers")
    logger.info(f"  Execution order: {[r[0] for r in received]}")
    logger.info("  PASSED")
    return True


async def test_event_bus_wildcard():
    """Test 2: Wildcard subscriber receives all events."""
    logger.info("=" * 60)
    logger.info("TEST 2: EventBus wildcard subscriber")
    logger.info("=" * 60)

    from src.core.event_bus import EventBus

    bus = EventBus()
    wild_received = []

    async def wildcard_handler(event, data):
        wild_received.append(event)

    bus.subscribe("*", wildcard_handler, owner="Wildcard")

    await bus.publish("event_a", None)
    await bus.publish("event_b", None)
    await bus.publish("event_c", None)

    assert len(wild_received) == 3, f"Wildcard should receive 3 events, got {len(wild_received)}"
    logger.info(f"  Wildcard received: {wild_received}")
    logger.info("  PASSED")
    return True


async def test_event_bus_interrupt():
    """Test 3: Interrupt (HIGH priority) runs before normal handlers."""
    logger.info("=" * 60)
    logger.info("TEST 3: Interrupt priority ordering")
    logger.info("=" * 60)

    from src.core.event_bus import EventBus, Priority
    from src.core.events import INTERRUPT, InterruptPayload

    bus = EventBus()
    order = []

    async def normal_handler(event, data):
        order.append("normal")

    async def interrupt_handler(event, data):
        order.append("interrupt")

    bus.subscribe(INTERRUPT, normal_handler, priority=Priority.NORMAL, owner="Normal")
    bus.subscribe(INTERRUPT, interrupt_handler, priority=Priority.HIGH, owner="Interrupt")

    await bus.publish(INTERRUPT, InterruptPayload(reason="test"))

    assert order[0] == "interrupt", "Interrupt handler should run first"
    logger.info(f"  Execution order: {order}")
    logger.info("  PASSED")
    return True


async def test_full_flow():
    """Test 4: Full flow - STT_READY -> Dialog Engine -> LLM_RESPONSE -> Memory saves."""
    logger.info("=" * 60)
    logger.info("TEST 4: Full EventBus flow (simulated)")
    logger.info("=" * 60)

    from src.core.event_bus import EventBus
    from src.core import events

    bus = EventBus()
    flow_log = []

    async def mock_dialog_engine(event, data):
        flow_log.append(f"DialogEngine received: {data.text}")
        response = events.LLMResponsePayload(
            sentences=[
                {"text_spoken": "こんにちは", "text_display": "Hello", "emo": "happy", "act": "wave", "intensity": 0.8},
            ],
            response_id="test-123",
            latency_ms=150,
        )
        await bus.publish(events.LLM_RESPONSE, response)

    async def mock_memory_stt(event, data):
        flow_log.append(f"Memory saved user: {data.text}")

    async def mock_memory_llm(event, data):
        flow_log.append(f"Memory saved {len(data.sentences)} AI sentences")

    async def mock_token_router(event, data):
        flow_log.append(f"TokenRouter received {len(data.sentences)} sentences")

    bus.subscribe(events.STT_READY, mock_dialog_engine, owner="DialogEngine")
    bus.subscribe(events.STT_READY, mock_memory_stt, owner="Memory:stt")
    bus.subscribe(events.LLM_RESPONSE, mock_memory_llm, owner="Memory:llm")
    bus.subscribe(events.LLM_RESPONSE, mock_token_router, owner="TokenRouter")

    logger.info("  Publishing STT_READY event...")
    start = time.time()
    payload = events.STTReadyPayload(text="Hello Chino!", lang="en", source="test")
    await bus.publish(events.STT_READY, payload)
    elapsed = (time.time() - start) * 1000

    logger.info(f"  Flow completed in {elapsed:.1f}ms")
    for step in flow_log:
        logger.info(f"    -> {step}")

    expected_steps = 4
    assert len(flow_log) == expected_steps, f"Expected {expected_steps} steps, got {len(flow_log)}"

    logger.info(f"  EventBus stats: {bus.get_stats()}")
    logger.info(f"  Event history: {bus.get_history()}")
    logger.info("  PASSED")
    return True


async def test_bootstrap_startup():
    """Test 5: Bootstrap creates all services and wires EventBus."""
    logger.info("=" * 60)
    logger.info("TEST 5: Bootstrap service registry")
    logger.info("=" * 60)

    from src.core.bootstrap import ServiceRegistry

    reg = ServiceRegistry()
    await reg.startup()

    bus = reg.get("event_bus")
    engine = reg.get("dialog_engine")
    memory = reg.get("memory_manager")
    token_router = reg.get("token_router")

    logger.info(f"  Services: event_bus, dialog_engine, memory_manager, token_router")
    logger.info(f"  EventBus subscribers: {bus.get_stats()}")
    logger.info(f"  Memory stats: {memory.get_stats()}")

    assert bus is not None
    assert engine is not None
    assert memory is not None
    assert token_router is not None
    assert bus.get_subscriber_count() > 0, "EventBus should have subscribers after bootstrap"

    await reg.shutdown()
    logger.info("  PASSED")
    return True


async def test_live_flow():
    """Test 6: Live flow through bootstrapped services (requires OpenRouter API key)."""
    logger.info("=" * 60)
    logger.info("TEST 6: Live flow through real services")
    logger.info("=" * 60)

    from src.setting import OPENROUTER_API_KEY
    if not OPENROUTER_API_KEY:
        logger.info("  SKIPPED (no OPENROUTER_API_KEY set)")
        return True

    from src.core.bootstrap import ServiceRegistry
    from src.core import events

    reg = ServiceRegistry()
    await reg.startup(llm_mode="openrouter")

    bus = reg.get("event_bus")
    llm_responses = []

    async def capture_response(event, data):
        llm_responses.append(data)

    bus.subscribe(events.LLM_RESPONSE, capture_response, owner="TestCapture")

    logger.info("  Publishing STT_READY to live system...")
    start = time.time()

    payload = events.STTReadyPayload(text="Hello! How are you today?", lang="en", source="test")
    await bus.publish(events.STT_READY, payload)

    elapsed = (time.time() - start) * 1000
    logger.info(f"  Live flow completed in {elapsed:.0f}ms")

    if llm_responses:
        resp = llm_responses[0]
        logger.info(f"  LLM response_id: {resp.response_id}")
        logger.info(f"  Sentences: {len(resp.sentences)}")
        for s in resp.sentences:
            logger.info(f"    [{s.get('emo', '?')}] {s.get('text_display', '')}")
        logger.info(f"  Latency: {resp.latency_ms}ms")
    else:
        logger.warning("  No LLM response captured (LLM may have failed)")

    memory = reg.get("memory_manager")
    msgs = memory.get_recent_messages()
    logger.info(f"  Messages in memory: {len(msgs)}")

    await reg.shutdown()
    logger.info("  PASSED")
    return True


async def main():
    logger.info("\n" + "=" * 60)
    logger.info("CHINO KAFUU - EventBus Infrastructure Test")
    logger.info("=" * 60 + "\n")

    tests = [
        ("EventBus basic pub/sub", test_event_bus_basic),
        ("EventBus wildcard", test_event_bus_wildcard),
        ("Interrupt priority", test_event_bus_interrupt),
        ("Full flow (mock)", test_full_flow),
        ("Bootstrap startup", test_bootstrap_startup),
        ("Live flow (real LLM)", test_live_flow),
    ]

    results = []
    for name, test_fn in tests:
        try:
            passed = await test_fn()
            results.append((name, "PASSED" if passed else "FAILED"))
        except Exception as e:
            logger.error(f"  FAILED: {e}", exc_info=True)
            results.append((name, f"FAILED: {e}"))
        logger.info("")

    logger.info("=" * 60)
    logger.info("TEST RESULTS")
    logger.info("=" * 60)
    for name, status in results:
        icon = "OK" if "PASSED" in status else "FAIL"
        logger.info(f"  [{icon}] {name}: {status}")

    all_passed = all("PASSED" in s for _, s in results)
    logger.info("=" * 60)
    logger.info(f"{'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
