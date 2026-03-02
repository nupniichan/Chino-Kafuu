import asyncio
import logging
import queue as thread_queue
from typing import Optional

import numpy as np

from src.core.event_bus import EventBus, Priority
from src.core import events
from src.setting import (
    STT_MODEL_PATH,
    VAD_THRESHOLD,
    SILENCE_CHUNKS_NEEDED,
    SAMPLE_RATE,
)

logger = logging.getLogger(__name__)

CHUNK_DURATION_MS = 100
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)


class RealtimePipeline:
    """
    Background loop: AudioCapture -> Queue -> Transcriber -> EventBus.

    Uses AudioCapture to bridge the sounddevice callback thread
    to the asyncio event loop. Each audio chunk is processed exactly once.
    """

    def __init__(self, event_bus: EventBus) -> None:
        self.bus = event_bus
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._chunk_queue: thread_queue.Queue = thread_queue.Queue(maxsize=300)
        self._audio_capture = None
        self._transcriber = None

    def _init_modules(self) -> None:
        from src.modules.asr.transcriber import Transcriber
        from src.modules.audio.capture import AudioCapture

        self._transcriber = Transcriber(
            vad_threshold=VAD_THRESHOLD,
            stt_model_path=STT_MODEL_PATH,
            sample_rate=SAMPLE_RATE,
            silence_chunks_needed=SILENCE_CHUNKS_NEEDED,
        )
        self._audio_capture = AudioCapture(
            on_chunk=self._on_audio_chunk,
            sample_rate=SAMPLE_RATE,
            block_size=CHUNK_SIZE,
        )
        logger.info("RealtimePipeline modules initialized")

    def _on_audio_chunk(self, chunk: np.ndarray) -> None:
        """Runs in audio thread — pushes float32 chunk into the async queue."""
        try:
            self._chunk_queue.put_nowait(chunk)
        except thread_queue.Full:
            pass

    def register(self) -> None:
        self.bus.subscribe(
            events.INTERRUPT,
            self._on_interrupt,
            priority=Priority.HIGH,
            owner="RealtimePipeline",
        )

    async def _on_interrupt(self, event: str, data: events.InterruptPayload) -> None:
        if self._transcriber:
            self._transcriber.reset()
        while not self._chunk_queue.empty():
            try:
                self._chunk_queue.get_nowait()
            except thread_queue.Empty:
                break
        logger.debug("Pipeline cleared on interrupt")

    async def start(self) -> None:
        if self._running:
            logger.warning("Pipeline already running")
            return

        self._init_modules()
        self._audio_capture.start()
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info("RealtimePipeline started")

    async def _loop(self) -> None:
        loop = asyncio.get_event_loop()

        while self._running:
            try:
                chunk = await loop.run_in_executor(
                    None, self._blocking_get_chunk
                )
                if chunk is None:
                    continue

                transcription = await asyncio.to_thread(
                    self._transcriber.process, chunk
                )

                if transcription and transcription.strip():
                    logger.info(f"Pipeline STT: '{transcription}'")
                    payload = events.STTReadyPayload(
                        text=transcription,
                        lang="ja",
                        source="mic",
                    )
                    await self.bus.publish(events.STT_READY, payload)

            except asyncio.CancelledError:
                logger.info("Pipeline loop cancelled")
                break
            except Exception as e:
                logger.error(f"Pipeline loop error: {e}", exc_info=True)
                await asyncio.sleep(0.5)

    def _blocking_get_chunk(self) -> Optional[np.ndarray]:
        """Blocking get from thread-safe queue (runs in executor thread)."""
        try:
            return self._chunk_queue.get(timeout=0.2)
        except thread_queue.Empty:
            return None

    def stop(self) -> None:
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
        if self._audio_capture:
            self._audio_capture.stop()
        logger.info("RealtimePipeline stopped")

    @property
    def is_running(self) -> bool:
        return self._running
