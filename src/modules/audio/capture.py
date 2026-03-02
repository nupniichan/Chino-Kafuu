import logging
import sounddevice as sd
import numpy as np
from typing import Optional, Any, Callable

LOG_INTERVAL_CHUNKS = 20


class AudioCapture:
    """Handles continuous microphone audio capture with real-time streaming."""

    def __init__(
        self,
        audio_buffer: Any = None,
        on_chunk: Optional[Callable[[np.ndarray], None]] = None,
        sample_rate: int = 16000,
        block_size: int = 1600,
    ) -> None:
        """
        Initialize audio capture.

        Args:
            audio_buffer: AudioBuffer instance for continuous storage (optional)
            on_chunk: Callback receiving each float32 chunk for real-time processing (optional)
            sample_rate: Audio sample rate in Hz
            block_size: Number of frames per callback (1600 @ 16kHz = 100ms)
        """
        self.logger = logging.getLogger(__name__)
        self.audio_buffer = audio_buffer
        self.on_chunk = on_chunk
        self.sample_rate: int = sample_rate
        self.block_size: int = block_size
        self.stream: Optional[sd.InputStream] = None
        self.callback_count: int = 0

    def _callback(
        self,
        indata: np.ndarray,
        frames: int,
        time: Any,
        status: sd.CallbackFlags,
    ) -> None:
        if status:
            self.logger.warning(f"Audio stream status: {status}")

        chunk = indata.flatten()

        if self.audio_buffer:
            self.audio_buffer.put(chunk)
        if self.on_chunk:
            self.on_chunk(chunk)

        self.callback_count += 1
        if self.callback_count % LOG_INTERVAL_CHUNKS == 0 and self.audio_buffer:
            duration = self.audio_buffer.get_duration()
            self.logger.debug(f"Microphone streaming: Buffer contains {duration:.1f}s of audio")

    def start(self) -> None:
        try:
            self.logger.info(f"Opening microphone stream at {self.sample_rate}Hz...")
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                blocksize=self.block_size,
                callback=self._callback,
                dtype="float32",
            )
            self.stream.start()
            self.logger.info("Microphone stream started successfully")
        except Exception as e:
            self.logger.error(f"Failed to start microphone stream: {e}", exc_info=True)
            raise

    def stop(self) -> None:
        if self.stream:
            self.logger.info("Stopping microphone stream...")
            try:
                self.stream.stop()
                self.stream.close()
                self.logger.info("Microphone stream closed")
            except Exception as e:
                self.logger.error(f"Error stopping microphone stream: {e}", exc_info=True)
