import torch
import logging
from typing import List, Optional

class VAD:
    """
    A class to process audio chunks and detect speech using Silero VAD.
    """
    def __init__(self, threshold: float = 0.5):
        """
        Initializes the Silero VAD model.
        """
        self.model, self.utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)
        self.threshold = threshold
        self.logger = logging.getLogger(__name__)

    def is_speech(self, chunk: torch.Tensor, sample_rate: int) -> bool:
        """
        Detects if there is speech in the given audio chunk.
        """
        if not isinstance(chunk, torch.Tensor):
            raise TypeError("Audio chunk must be a torch.Tensor")

        speech_prob = self.model(chunk, sample_rate).item()
        is_speech = speech_prob >= self.threshold
        self.logger.debug(f"Speech probability: {speech_prob:.2f}, Is speech: {is_speech}")
        return is_speech

    def reset_states(self):
        """
        Resets the VAD model states.
        """
        self.logger.info("Resetting VAD states.")
        self.model.reset_states()