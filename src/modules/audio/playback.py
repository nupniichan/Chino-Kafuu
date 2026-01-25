"""
Audio playback module for playing synthesized audio files.
Supports both synchronous and asynchronous playback.
"""

import logging
import asyncio
import os
from pathlib import Path
from typing import Optional
import threading

logger = logging.getLogger(__name__)


class AudioPlayer:
    """Handles audio playback with support for various formats."""

    def __init__(self):
        self._playing = False
        self._current_thread: Optional[threading.Thread] = None

    def play(self, audio_path: str, blocking: bool = False) -> bool:
        """
        Play an audio file.
        
        Args:
            audio_path: Path to the audio file to play
            blocking: If True, blocks until playback completes
            
        Returns:
            True if playback started successfully, False otherwise
        """
        if not audio_path or not Path(audio_path).exists():
            logger.error(f"Audio file not found: {audio_path}")
            return False

        try:
            # Try pygame first (lightweight, cross-platform)
            return self._play_with_pygame(audio_path, blocking)
        except ImportError:
            try:
                # Fallback to playsound
                return self._play_with_playsound(audio_path, blocking)
            except ImportError:
                try:
                    # Fallback to pydub + simpleaudio
                    return self._play_with_pydub(audio_path, blocking)
                except ImportError:
                    logger.error(
                        "No audio playback library available. "
                        "Please install: pip install pygame or pip install playsound or pip install pydub simpleaudio"
                    )
                    return False

    def _play_with_pygame(self, audio_path: str, blocking: bool) -> bool:
        """Play audio using pygame mixer."""
        import pygame
        
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            
            pygame.mixer.music.load(audio_path)
            pygame.mixer.music.play()
            self._playing = True
            
            logger.info(f"Playing audio: {audio_path}")
            
            if blocking:
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)
                self._playing = False
            
            return True
            
        except Exception as e:
            logger.error(f"Error playing audio with pygame: {e}")
            return False

    def _play_with_playsound(self, audio_path: str, blocking: bool) -> bool:
        """Play audio using playsound library."""
        from playsound import playsound
        
        try:
            if blocking:
                playsound(audio_path)
                logger.info(f"Played audio: {audio_path}")
            else:
                def play_async():
                    playsound(audio_path)
                    self._playing = False
                    logger.info(f"Finished playing: {audio_path}")
                
                self._current_thread = threading.Thread(target=play_async, daemon=True)
                self._current_thread.start()
                logger.info(f"Started playing audio: {audio_path}")
            
            self._playing = True
            return True
            
        except Exception as e:
            logger.error(f"Error playing audio with playsound: {e}")
            return False

    def _play_with_pydub(self, audio_path: str, blocking: bool) -> bool:
        """Play audio using pydub + simpleaudio."""
        from pydub import AudioSegment
        from pydub.playback import play
        import simpleaudio as sa
        
        try:
            audio = AudioSegment.from_file(audio_path)
            
            if blocking:
                play(audio)
                logger.info(f"Played audio: {audio_path}")
            else:
                def play_async():
                    play(audio)
                    self._playing = False
                    logger.info(f"Finished playing: {audio_path}")
                
                self._current_thread = threading.Thread(target=play_async, daemon=True)
                self._current_thread.start()
                logger.info(f"Started playing audio: {audio_path}")
            
            self._playing = True
            return True
            
        except Exception as e:
            logger.error(f"Error playing audio with pydub: {e}")
            return False

    async def play_async(self, audio_path: str) -> bool:
        """
        Play audio asynchronously (non-blocking).
        
        Args:
            audio_path: Path to the audio file to play
            
        Returns:
            True if playback started successfully, False otherwise
        """
        return await asyncio.to_thread(self.play, audio_path, blocking=False)

    def stop(self) -> None:
        """Stop current playback."""
        try:
            import pygame
            if pygame.mixer.get_init():
                pygame.mixer.music.stop()
        except ImportError:
            pass
        
        self._playing = False
        logger.info("Stopped audio playback")

    def is_playing(self) -> bool:
        """Check if audio is currently playing."""
        try:
            import pygame
            if pygame.mixer.get_init():
                return pygame.mixer.music.get_busy()
        except ImportError:
            pass
        
        return self._playing

    def wait_until_done(self, timeout: Optional[float] = None) -> bool:
        """
        Wait until current playback is done.
        
        Args:
            timeout: Maximum time to wait in seconds (None = wait indefinitely)
            
        Returns:
            True if playback finished normally, False if timed out
        """
        try:
            import pygame
            if pygame.mixer.get_init():
                import time
                start_time = time.time()
                while pygame.mixer.music.get_busy():
                    if timeout and (time.time() - start_time) > timeout:
                        return False
                    pygame.time.Clock().tick(10)
                return True
        except ImportError:
            pass
        
        if self._current_thread:
            self._current_thread.join(timeout=timeout)
            return not self._current_thread.is_alive()
        
        return True


# Singleton instance for convenience
_default_player = AudioPlayer()


def play_audio(audio_path: str, blocking: bool = False) -> bool:
    """
    Convenience function to play audio using the default player.
    
    Args:
        audio_path: Path to the audio file to play
        blocking: If True, blocks until playback completes
        
    Returns:
        True if playback started successfully, False otherwise
    """
    return _default_player.play(audio_path, blocking)


async def play_audio_async(audio_path: str) -> bool:
    """
    Convenience function to play audio asynchronously.
    
    Args:
        audio_path: Path to the audio file to play
        
    Returns:
        True if playback started successfully, False otherwise
    """
    return await _default_player.play_async(audio_path)


def stop_audio() -> None:
    """Stop current playback."""
    _default_player.stop()


def is_audio_playing() -> bool:
    """Check if audio is currently playing."""
    return _default_player.is_playing()