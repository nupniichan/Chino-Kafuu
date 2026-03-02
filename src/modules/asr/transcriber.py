import logging
import torch
import numpy as np
from typing import Optional
from .vad import VAD
from .stt import STT


class Transcriber:
    """Wraps VAD and STT to provide a streaming transcription service."""

    def __init__(
        self,
        vad_threshold: float = 0.5,
        stt_model_path: str = "models/faster-whisper-small",
        sample_rate: int = 16000,
        silence_chunks_needed: int = 5,
    ):
        self.logger = logging.getLogger(__name__)
        self.sample_rate = sample_rate
        self.vad = VAD(threshold=vad_threshold)
        self.stt = STT(model_path=stt_model_path)

        self._speech_buffer: list[np.ndarray] = []
        self.is_speaking = False
        self.silence_chunks_counter = 0
        self.silence_chunks_needed = silence_chunks_needed

    def process(self, audio_chunk: np.ndarray) -> Optional[str]:
        """
        Process a float32 audio chunk, perform VAD and STT.
        Returns transcription if a speech segment completed, None otherwise.
        """
        audio_tensor = torch.from_numpy(audio_chunk).to(torch.float32)

        if self.vad.is_speech(audio_tensor, self.sample_rate):
            self.logger.debug("Speech detected.")
            self.is_speaking = True
            self.silence_chunks_counter = 0
            self._speech_buffer.append(audio_chunk)
        else:
            self.logger.debug("Silence detected.")
            if self.is_speaking:
                self.silence_chunks_counter += 1
                if self.silence_chunks_counter >= self.silence_chunks_needed:
                    self.logger.info(
                        f"End of speech detected after {self.silence_chunks_needed} silence chunks."
                    )

                    speech_segment = np.concatenate(self._speech_buffer)
                    transcription = self.stt.transcribe(speech_segment)

                    self.reset()
                    self.vad.reset_states()
                    return transcription

        return None

    def reset(self) -> None:
        """Reset speech detection state."""
        self._speech_buffer.clear()
        self.is_speaking = False
        self.silence_chunks_counter = 0
