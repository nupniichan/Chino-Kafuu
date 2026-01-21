import asyncio
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional

from modules.memory.token_counter import TokenCounter


@dataclass(frozen=True, slots=True)
class RoutedSentence:
    sequence_id: int
    queue_label: str
    token_count: int
    sentence: Dict[str, Any]


LabelSelector = Callable[[Dict[str, Any]], str]
TokenTextSelector = Callable[[Dict[str, Any]], str]


class TokenRouter:
    def __init__(
        self,
        token_counter: TokenCounter,
        *,
        default_queue_label: str = "default",
        token_text_selector: Optional[TokenTextSelector] = None,
    ) -> None:
        self._token_counter = token_counter
        self._default_queue_label = default_queue_label
        self._token_text_selector = token_text_selector or (lambda s: str(s.get("text_spoken", "") or ""))
        self._queues: Dict[str, asyncio.Queue[RoutedSentence]] = {}
        self._next_sequence_id = 0

    def get_queue(self, label: str) -> "asyncio.Queue[RoutedSentence]":
        queue = self._queues.get(label)
        if queue is None:
            queue = asyncio.Queue()
            self._queues[label] = queue
        return queue

    def route(
        self,
        sentences: Iterable[Dict[str, Any]],
        *,
        label_selector: Optional[LabelSelector] = None,
    ) -> List[RoutedSentence]:
        selector = label_selector or (lambda _: self._default_queue_label)

        routed: List[RoutedSentence] = []
        for sentence in sentences:
            label = selector(sentence) or self._default_queue_label
            text_for_tokens = self._token_text_selector(sentence)
            token_count = self._token_counter.count_tokens(text_for_tokens)

            item = RoutedSentence(
                sequence_id=self._next_sequence_id,
                queue_label=label,
                token_count=token_count,
                sentence=sentence,
            )
            self._next_sequence_id += 1

            self.get_queue(label).put_nowait(item)
            routed.append(item)

        return routed

    @staticmethod
    def reorder(routed: Iterable[RoutedSentence]) -> List[RoutedSentence]:
        items = list(routed)
        items.sort(key=lambda x: x.sequence_id)
        return items

    async def collect_in_order(
        self,
        sources: Iterable["asyncio.Queue[RoutedSentence]"],
        *,
        expected_count: int,
        start_sequence_id: int = 0,
    ) -> List[RoutedSentence]:
        queues = list(sources)
        if expected_count <= 0:
            return []

        buffer: Dict[int, RoutedSentence] = {}
        next_id = start_sequence_id
        out: List[RoutedSentence] = []

        while len(out) < expected_count:
            get_tasks = [asyncio.create_task(q.get()) for q in queues]
            done, pending = await asyncio.wait(get_tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()

            for task in done:
                item = task.result()
                buffer[item.sequence_id] = item

            while next_id in buffer and len(out) < expected_count:
                out.append(buffer.pop(next_id))
                next_id += 1

        return out