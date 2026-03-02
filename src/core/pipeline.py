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
    Background loop: Mic -> Queue -> Transcriber -> EventBus.

    Uses a thread-safe queue to bridge the sounddevice callback thread
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

        self._transcriber = Transcriber(
            vad_threshold=VAD_THRESHOLD,
            stt_model_path=STT_MODEL_PATH,
            sample_rate=SAMPLE_RATE,
            silence_chunks_needed=SILENCE_CHUNKS_NEEDED,
        )
        logger.info("RealtimePipeline modules initialized")

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        """Sounddevice callback -- runs in audio thread, puts bytes into queue."""
        if status:
            logger.warning(f"Audio stream status: {status}")
        chunk_bytes = (indata.flatten() * 32768.0).astype(np.int16).tobytes()
        try:
            self._chunk_queue.put_nowait(chunk_bytes)
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
            self._transcriber.audio_buffer.clear()
            self._transcriber.is_speaking = False
            self._transcriber.silence_chunks_counter = 0
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
        self._start_audio_stream()
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info("RealtimePipeline started")

    def _start_audio_stream(self) -> None:
        import sounddevice as sd

        self._audio_stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            blocksize=CHUNK_SIZE,
            callback=self._audio_callback,
            dtype="float32",
        )
        self._audio_stream.start()
        logger.info(f"Mic stream opened at {SAMPLE_RATE}Hz, chunk={CHUNK_SIZE} frames")

    async def _loop(self) -> None:
        loop = asyncio.get_event_loop()

        while self._running:
            try:
                chunk_bytes = await loop.run_in_executor(
                    None, self._blocking_get_chunk
                )
                if chunk_bytes is None:
                    continue

                transcription = await asyncio.to_thread(
                    self._transcriber.process, chunk_bytes
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

    def _blocking_get_chunk(self) -> Optional[bytes]:
        """Blocking get from thread-safe queue (runs in executor thread)."""
        try:
            return self._chunk_queue.get(timeout=0.2)
        except thread_queue.Empty:
            return None

    def stop(self) -> None:
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
        if hasattr(self, "_audio_stream") and self._audio_stream:
            try:
                self._audio_stream.stop()
                self._audio_stream.close()
            except Exception:
                pass
        logger.info("RealtimePipeline stopped")

    @property
    def is_running(self) -> bool:
        return self._running
