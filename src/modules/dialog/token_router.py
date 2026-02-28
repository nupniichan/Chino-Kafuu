"""
Token Router: Manages sentence ordering and FIFO routing for LLM responses.

Pipeline: LLM Response → Process Response → TokenRouter → [Stream/TTS/RVC] → Build Final

The router ensures sentences are:
1. Labeled with indices to prevent misordering
2. Queued in FIFO order per slot for parallel processing
3. Tracked for completion status
4. Returned in correct order when all processing is done
"""
import asyncio
import logging
import uuid
from typing import Dict, Any, List, Optional, AsyncIterator
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class SentenceStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"


@dataclass
class LabeledSentence:
    """A sentence labeled with index for ordering prevention."""
    index: int
    data: Dict[str, Any]
    status: SentenceStatus = SentenceStatus.PENDING
    response_id: str = ""


class SentenceFIFO:
    """FIFO queue for a single router slot. Each slot processes sentences sequentially."""

    def __init__(self, slot_id: int):
        self.slot_id = slot_id
        self._queue: asyncio.Queue[LabeledSentence] = asyncio.Queue()
        self._current: Optional[LabeledSentence] = None

    async def enqueue(self, sentence: LabeledSentence):
        await self._queue.put(sentence)

    async def dequeue(self) -> Optional[LabeledSentence]:
        try:
            self._current = self._queue.get_nowait()
            self._current.status = SentenceStatus.PROCESSING
            return self._current
        except asyncio.QueueEmpty:
            return None

    async def wait_next(self) -> LabeledSentence:
        """Block until a sentence is available (for streaming consumers)."""
        self._current = await self._queue.get()
        self._current.status = SentenceStatus.PROCESSING
        return self._current

    def complete_current(self) -> Optional[LabeledSentence]:
        if self._current is None:
            return None
        self._current.status = SentenceStatus.COMPLETED
        completed = self._current
        self._current = None
        return completed

    @property
    def is_idle(self) -> bool:
        return self._current is None and self._queue.empty()

    @property
    def pending_count(self) -> int:
        return self._queue.qsize()


class TokenRouter:
    """
    Routes labeled sentences through parallel FIFO slots.

    Distributes sentences round-robin across slots. Each slot processes
    its sentences in FIFO order. Completed sentences are collected
    and can be reassembled in the original order.

    Usage (non-streaming):
        router = TokenRouter(num_slots=2)
        labeled = await router.route_sentences(parsed_sentences)
        # Process each slot's sentences...
        for slot_id in range(router.num_slots):
            while sentence := await router.get_next(slot_id):
                # process sentence (TTS, RVC, etc.)
                router.mark_completed(sentence.index)
        result = router.build_ordered_response()

    Usage (streaming):
        router = TokenRouter(num_slots=2)
        await router.route_sentences(parsed_sentences)
        async for sentence in router.iter_slot(slot_id):
            # stream tokens for this sentence
    """

    def __init__(self, num_slots: int = 2):
        self.num_slots = num_slots
        self._slots: List[SentenceFIFO] = [SentenceFIFO(i) for i in range(num_slots)]
        self._completed: Dict[int, LabeledSentence] = {}
        self._all_sentences: Dict[int, LabeledSentence] = {}
        self._total: int = 0
        self._response_id: str = ""
        self._lock = asyncio.Lock()
        self._finished_event = asyncio.Event()

    async def route_sentences(
        self,
        sentences: List[Dict[str, Any]],
        response_id: Optional[str] = None
    ) -> List[LabeledSentence]:
        """Label sentences and distribute to FIFO slots round-robin."""
        async with self._lock:
            self.reset()
            self._response_id = response_id or str(uuid.uuid4())
            self._total = len(sentences)

            labeled = []
            for idx, sentence_data in enumerate(sentences):
                sentence = LabeledSentence(
                    index=idx,
                    data=sentence_data,
                    response_id=self._response_id
                )
                self._all_sentences[idx] = sentence
                labeled.append(sentence)

                slot = self._slots[idx % self.num_slots]
                await slot.enqueue(sentence)
                logger.debug(f"Sentence #{idx} → slot {slot.slot_id}")

            logger.info(
                f"Routed {self._total} sentences across {self.num_slots} slots "
                f"(response_id={self._response_id})"
            )
            return labeled

    async def get_next(self, slot_id: int) -> Optional[LabeledSentence]:
        """Get next pending sentence from a slot (non-blocking)."""
        if slot_id >= self.num_slots:
            return None
        return await self._slots[slot_id].dequeue()

    async def wait_next(self, slot_id: int) -> LabeledSentence:
        """Wait for next sentence from a slot (blocking, for streaming)."""
        return await self._slots[slot_id].wait_next()

    def mark_completed(self, sentence_index: int) -> bool:
        """Mark a sentence as completed after processing (TTS/RVC done)."""
        if sentence_index not in self._all_sentences:
            logger.warning(f"Unknown sentence index: {sentence_index}")
            return False

        sentence = self._all_sentences[sentence_index]
        if sentence.status == SentenceStatus.COMPLETED:
            return False

        slot = self._slots[sentence_index % self.num_slots]
        slot.complete_current()

        self._completed[sentence_index] = sentence
        logger.debug(f"Sentence #{sentence_index} completed ({len(self._completed)}/{self._total})")

        if self.is_all_completed:
            self._finished_event.set()

        return True

    @property
    def is_all_completed(self) -> bool:
        return self._total > 0 and len(self._completed) == self._total

    async def wait_until_done(self, timeout: Optional[float] = None) -> bool:
        """Wait until all sentences are completed. Returns False on timeout."""
        try:
            await asyncio.wait_for(self._finished_event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    def build_ordered_response(self) -> List[Dict[str, Any]]:
        """Reassemble all completed sentences in original index order."""
        ordered = sorted(self._completed.values(), key=lambda s: s.index)
        return [s.data for s in ordered]

    def get_completed_up_to(self) -> List[Dict[str, Any]]:
        """Get contiguous completed sentences from index 0 (for partial returns)."""
        result = []
        for i in range(self._total):
            if i not in self._completed:
                break
            result.append(self._completed[i].data)
        return result

    def reset(self):
        """Reset router state for a new response."""
        self._slots = [SentenceFIFO(i) for i in range(self.num_slots)]
        self._completed.clear()
        self._all_sentences.clear()
        self._total = 0
        self._response_id = ""
        self._finished_event.clear()

    def get_status(self) -> Dict[str, Any]:
        return {
            "response_id": self._response_id,
            "total_sentences": self._total,
            "completed": len(self._completed),
            "is_all_completed": self.is_all_completed,
            "slots": [
                {
                    "slot_id": slot.slot_id,
                    "pending": slot.pending_count,
                    "is_idle": slot.is_idle
                }
                for slot in self._slots
            ]
        }

    async def process_all_sequential(self) -> List[Dict[str, Any]]:
        """
        Process all sentences across all slots sequentially (non-streaming).
        Marks each sentence as completed and returns the ordered response.
        Used as a simple pass-through before streaming is integrated.
        """
        for slot_id in range(self.num_slots):
            while True:
                sentence = await self.get_next(slot_id)
                if sentence is None:
                    break
                self.mark_completed(sentence.index)

        return self.build_ordered_response()