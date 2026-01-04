import logging
import torch
import numpy as np
from collections import deque
from typing import Optional
from .vad import VAD
from .stt import STT

class Transcriber:
    """
    A class that wraps VAD and STT to provide a streaming transcription service.
    """
    def __init__(self, 
                 vad_threshold: float = 0.5, 
                 stt_model_path: str = "models/faster-whisper-small",
                 sample_rate: int = 16000,
                 silence_chunks_needed: int = 5):
        self.logger = logging.getLogger(__name__)
        self.sample_rate = sample_rate
        self.vad = VAD(threshold=vad_threshold)
        self.stt = STT(model_path=stt_model_path)
        
        self.audio_buffer = deque()
        self.is_speaking = False
        self.silence_chunks_counter = 0
        self.silence_chunks_needed = silence_chunks_needed

    @staticmethod
    def _bytes_to_float_tensor(chunk: bytes) -> torch.Tensor:
        """Converts raw audio bytes to a float tensor."""
        audio_int16 = np.frombuffer(chunk, dtype=np.int16).copy()
        return torch.from_numpy(audio_int16).to(torch.float32) / 32768.0

    def process(self, chunk: bytes) -> Optional[str]:
        """
        Processes a chunk of audio, performs VAD and STT, and returns a transcription if available.
        """
        transcription = None
        audio_tensor = self._bytes_to_float_tensor(chunk)
        
        if self.vad.is_speech(audio_tensor, self.sample_rate):
            self.logger.debug("Speech detected.")
            self.is_speaking = True
            self.silence_chunks_counter = 0
            self.audio_buffer.append(chunk)
        else:
            self.logger.debug("Silence detected.")
            if self.is_speaking:
                self.silence_chunks_counter += 1
                if self.silence_chunks_counter >= self.silence_chunks_needed:
                    self.logger.info(f"End of speech detected after {self.silence_chunks_needed} silence chunks.")
                    
                    speech_segment_bytes = b"".join(list(self.audio_buffer))
                    speech_segment_numpy = np.frombuffer(speech_segment_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                    
                    transcription = self.stt.transcribe(speech_segment_numpy)
                    
                    self.audio_buffer.clear()
                    self.is_speaking = False
                    self.silence_chunks_counter = 0
                    self.vad.reset_states()
        
        return transcription