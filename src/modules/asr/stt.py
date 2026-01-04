import logging
import numpy as np
from typing import Optional
from faster_whisper import WhisperModel

class STT:
    """
    Speech-to-Text service using the Faster-Whisper model.
    
    Supports both CPU and GPU inference with configurable compute types.
    For CPU usage: device="cpu", compute_type="int8"
    For GPU usage: device="cuda", compute_type="float16"
    """
    
    def __init__(
        self, 
        model_path: str = "models/faster-whisper-small", 
        device: str = "cuda", 
        compute_type: str = "float16"
    ) -> None:
        """
        Initialize the STT model.
        
        Args:
            model_path: Path to the Faster-Whisper model directory
            device: Device to run inference on ("cuda" or "cpu")
            compute_type: Compute precision ("float16", "int8", etc.)
        
        Raises:
            Exception: If model loading fails
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Loading Faster-Whisper model from: {model_path}")
        try:
            self.model = WhisperModel(model_path, device=device, compute_type=compute_type)
            self.logger.info("Faster-Whisper model loaded successfully.")
        except Exception as e:
            self.logger.error(f"Failed to load Faster-Whisper model: {e}", exc_info=True)
            raise

    def transcribe(
        self, 
        audio_data: np.ndarray, 
        beam_size: int = 5, 
        language: Optional[str] = None
    ) -> str:
        """
        Transcribe audio data to text.
        
        Args:
            audio_data: Audio data as float32 numpy array
            beam_size: Beam search size for decoding (higher = more accurate but slower)
            language: Force specific language (None = auto-detect)
        
        Returns:
            Transcribed text as string
        
        Raises:
            TypeError: If audio_data is not a numpy array
        """
        if not isinstance(audio_data, np.ndarray):
            raise TypeError("Audio data must be a numpy.ndarray")

        self.logger.debug(f"Transcribing audio data of shape: {audio_data.shape}")
        
        segments, info = self.model.transcribe(
            audio_data, 
            beam_size=beam_size,
            language=language
        )
        
        transcription = "".join(segment.text for segment in segments).strip()
        
        if transcription:
            self.logger.info(f"Transcription: '{transcription}'")
        else:
            self.logger.debug("Transcription is empty.")
            
        self.logger.debug(f"Detected language: {info.language} with probability {info.language_probability:.2f}")
        
        return transcription