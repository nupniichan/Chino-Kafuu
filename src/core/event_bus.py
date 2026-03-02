"""
Async EventBus: Central communication backbone for the Chino Kafuu AI system.

Lightweight pub/sub built on asyncio. Supports priority handlers (for interrupts),
wildcard subscribers (for logging/debug), and concurrent handler execution.
"""
import asyncio
import logging
from typing import Any, Callable, Coroutine, Dict, List, Optional
from dataclasses import dataclass, field
from enum import IntEnum

logger = logging.getLogger(__name__)

HandlerFn = Callable[..., Coroutine[Any, Any, None]]


class Priority(IntEnum):
    HIGH = 0
    NORMAL = 1
    LOW = 2


@dataclass(order=True)
class Subscription:
    priority: Priority
    handler: HandlerFn = field(compare=False)
    owner: str = field(default="", compare=False)


class EventBus:
    """
    Async event bus using dict-based.

    Usage:
        bus = EventBus()
        bus.subscribe("stt_ready", my_handler)
        await bus.publish("stt_ready", {"text": "hello"})
    """

    def __init__(self) -> None:
        self._subscribers: Dict[str, List[Subscription]] = {}
        self._lock = asyncio.Lock()
        self._history: List[Dict[str, Any]] = []
        self._max_history = 100

    def subscribe(
        self,
        event: str,
        handler: HandlerFn,
        priority: Priority = Priority.NORMAL,
        owner: str = "",
    ) -> None:
        if event not in self._subscribers:
            self._subscribers[event] = []

        sub = Subscription(priority=priority, handler=handler, owner=owner)
        self._subscribers[event].append(sub)
        self._subscribers[event].sort()

        logger.debug(f"[EventBus] {owner or handler.__name__} subscribed to '{event}' (priority={priority.name})")

    def unsubscribe(self, event: str, handler: HandlerFn) -> bool:
        if event not in self._subscribers:
            return False

        before = len(self._subscribers[event])
        self._subscribers[event] = [s for s in self._subscribers[event] if s.handler is not handler]
        removed = len(self._subscribers[event]) < before

        if not self._subscribers[event]:
            del self._subscribers[event]

        return removed

    def unsubscribe_all(self, owner: str) -> int:
        removed = 0
        empty_events = []

        for event, subs in self._subscribers.items():
            before = len(subs)
            self._subscribers[event] = [s for s in subs if s.owner != owner]
            removed += before - len(self._subscribers[event])
            if not self._subscribers[event]:
                empty_events.append(event)

        for event in empty_events:
            del self._subscribers[event]

        return removed

    async def publish(self, event: str, data: Any = None) -> int:
        handlers = self._get_handlers(event)
        if not handlers:
            logger.debug(f"[EventBus] No handlers for '{event}'")
            return 0

        self._record_history(event, data)

        tasks = []
        for sub in handlers:
            tasks.append(self._safe_call(sub, event, data))

        await asyncio.gather(*tasks)
        return len(tasks)

    async def publish_and_wait(self, event: str, data: Any = None, timeout: float = 30.0) -> int:
        try:
            return await asyncio.wait_for(self.publish(event, data), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(f"[EventBus] Publish '{event}' timed out after {timeout}s")
            return 0

    def _get_handlers(self, event: str) -> List[Subscription]:
        handlers = list(self._subscribers.get(event, []))
        wildcard = self._subscribers.get("*", [])
        if wildcard:
            handlers.extend(wildcard)
            handlers.sort()
        return handlers

    async def _safe_call(self, sub: Subscription, event: str, data: Any) -> None:
        try:
            await sub.handler(event, data)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(
                f"[EventBus] Handler '{sub.owner or sub.handler.__name__}' "
                f"failed on '{event}': {e}",
                exc_info=True,
            )

    def _record_history(self, event: str, data: Any) -> None:
        import time
        self._history.append({"event": event, "timestamp": time.time(), "data_type": type(data).__name__})
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

    def get_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        return self._history[-limit:]

    def get_subscriber_count(self, event: Optional[str] = None) -> int:
        if event:
            return len(self._subscribers.get(event, []))
        return sum(len(subs) for subs in self._subscribers.values())

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_subscribers": self.get_subscriber_count(),
            "events": {
                event: len(subs) for event, subs in self._subscribers.items()
            },
            "history_size": len(self._history),
        }

    def reset(self) -> None:
        self._subscribers.clear()
        self._history.clear()
        logger.info("[EventBus] Reset complete")
