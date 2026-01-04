import numpy as np
import threading
from typing import Union

class AudioBuffer:
    """
    High-performance circular buffer for audio data using pre-allocated numpy arrays.
    Thread-safe for concurrent read/write operations.
    """
    
    def __init__(self, max_seconds: int = 30, sample_rate: int = 16000) -> None:
        """
        Initialize the audio buffer.
        
        Args:
            max_seconds: Maximum duration of audio to store in seconds
            sample_rate: Audio sample rate in Hz
        """
        self.sample_rate: int = sample_rate
        self.max_frames: int = int(max_seconds * sample_rate)
        
        self.buffer: np.ndarray = np.zeros(self.max_frames, dtype=np.float32)
        self.write_index: int = 0
        self.total_frames_written: int = 0
        
        self.lock: threading.Lock = threading.Lock()

    def put(self, data: Union[np.ndarray, bytes]) -> None:
        """Write new audio data into the buffer.
        
        Args:
            data: Audio data as numpy array or bytes
        """
        with self.lock:
            if len(data.shape) > 1:
                data = data.flatten()
            
            num_frames = len(data)
            if num_frames > self.max_frames:
                data = data[-self.max_frames:]
                num_frames = self.max_frames

            # Calculate how much space is left before we hit the end of the array
            end_space = self.max_frames - self.write_index
            
            # If it fits perfectly 
            if num_frames <= end_space:
                self.buffer[self.write_index:self.write_index + num_frames] = data

            # If not then we wrap it fill the end and put the rest at the start
            else:
                self.buffer[self.write_index:] = data[:end_space]
                self.buffer[:num_frames - end_space] = data[end_space:]
            
            # Move the pointer and keep track of total frames for duration calc
            self.write_index = (self.write_index + num_frames) % self.max_frames
            self.total_frames_written += num_frames

    def get_last_n_seconds(self, seconds: float) -> np.ndarray:
        """Retrieve the most recent N seconds of audio without clearing the buffer.
        
        Args:
            seconds: Number of seconds of audio to retrieve
        
        Returns:
            Audio data as numpy array
        """
        num_frames = int(seconds * self.sample_rate)
        return self._get_frames(num_frames)

    def _get_frames(self, num_frames: int) -> np.ndarray:
        with self.lock:
            available_frames = min(self.total_frames_written, self.max_frames)
            frames_to_get = min(num_frames, available_frames)
            
            if frames_to_get == 0:
                return np.array([], dtype=np.float32)

            start_index = (self.write_index - frames_to_get) % self.max_frames
            
            if start_index + frames_to_get <= self.max_frames:
                return self.buffer[start_index:start_index + frames_to_get].copy()
            else:
                part1 = self.buffer[start_index:]
                part2 = self.buffer[:frames_to_get - len(part1)]
                return np.concatenate([part1, part2])

    def clear(self) -> None:
        """Reset the buffer pointers and clear all data."""
        with self.lock:
            self.buffer.fill(0)
            self.write_index = 0
            self.total_frames_written = 0

    def get_duration(self) -> float:
        """Get the current duration of audio stored in the buffer.
        
        Returns:
            Duration in seconds
        """
        return min(self.total_frames_written, self.max_frames) / self.sample_rate