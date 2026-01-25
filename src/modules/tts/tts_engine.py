"""
Text-to-Speech engine that integrates RVC conversion and audio playback.
"""

import logging
from pathlib import Path
from typing import Optional
import asyncio

from .rvc_converter import RvcConverter, RvcEnforceTerms2Request, RvcEnforceTerms2Result
from ..audio.playback import AudioPlayer, play_audio, play_audio_async

logger = logging.getLogger(__name__)


class TTSEngine:
    """
    Complete TTS engine that converts text to speech using RVC and plays the result.
    """

    def __init__(
        self,
        rvc_base_url: str = "http://127.0.0.1:6969/",
        default_voice_model: str = "Chino-Kafuu",
        default_index_file: Optional[str] = None,
        output_dir: str = r"C:\Users\nup\Downloads\Applio\assets\audios",
    ):
        """
        Initialize TTS engine.
        
        Args:
            rvc_base_url: URL of the RVC Gradio server
            default_voice_model: Default voice model to use
            default_index_file: Default index file path
            output_dir: Directory to save output audio files
        """
        self.rvc = RvcConverter(base_url=rvc_base_url)
        self.player = AudioPlayer()
        self.default_voice_model = default_voice_model
        self.default_index_file = default_index_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def synthesize(
        self,
        text: str,
        voice_model: Optional[str] = None,
        play_audio_after: bool = True,
        blocking_playback: bool = False,
        **kwargs,
    ) -> RvcEnforceTerms2Result:
        """
        Synthesize text to speech and optionally play it.
        
        Args:
            text: Text to synthesize
            voice_model: Voice model to use (defaults to default_voice_model)
            play_audio_after: Whether to play audio after synthesis
            blocking_playback: If True, blocks until playback completes
            **kwargs: Additional parameters for RvcEnforceTerms2Request
            
        Returns:
            RvcEnforceTerms2Result containing output information and audio path
        """
        if not text:
            raise ValueError("Text to synthesize cannot be empty")

        # Prepare request
        request_kwargs = {
            "terms_accepted": False,
            "text_to_synthesize": text,
            "voice_model": voice_model or self.default_voice_model,
            "index_file": self.default_index_file,
            "output_path_tts_audio": str(self.output_dir / "tts_output.wav"),
            "output_path_rvc_audio": str(self.output_dir / "tts_rvc_output.wav"),
        }
        request_kwargs.update(kwargs)

        request = RvcEnforceTerms2Request(**request_kwargs)
        
        logger.info(f"Synthesizing text: {text[:50]}...")
        result = self.rvc.enforce_terms_2(request)
        
        logger.info(f"Synthesis complete. Audio saved to: {result.export_audio_path}")

        if play_audio_after and result.export_audio_path:
            success = self.player.play(result.export_audio_path, blocking=blocking_playback)
            if success:
                logger.info(f"Playing audio: {result.export_audio_path}")
            else:
                logger.warning("Failed to play audio")

        return result

    async def synthesize_async(
        self,
        text: str,
        voice_model: Optional[str] = None,
        play_audio_after: bool = True,
        **kwargs,
    ) -> RvcEnforceTerms2Result:
        """
        Asynchronously synthesize text to speech and play it.
        
        Args:
            text: Text to synthesize
            voice_model: Voice model to use (defaults to default_voice_model)
            play_audio_after: Whether to play audio after synthesis
            **kwargs: Additional parameters for RvcEnforceTerms2Request
            
        Returns:
            RvcEnforceTerms2Result containing output information and audio path
        """
        if not text:
            raise ValueError("Text to synthesize cannot be empty")

        # Prepare request
        request_kwargs = {
            "terms_accepted": False,
            "text_to_synthesize": text,
            "voice_model": voice_model or self.default_voice_model,
            "index_file": self.default_index_file,
            "output_path_tts_audio": str(self.output_dir / "tts_output.wav"),
            "output_path_rvc_audio": str(self.output_dir / "tts_rvc_output.wav"),
        }
        request_kwargs.update(kwargs)

        request = RvcEnforceTerms2Request(**request_kwargs)
        
        logger.info(f"Synthesizing text asynchronously: {text[:50]}...")
        result = await self.rvc.enforce_terms_2_async(request)
        
        logger.info(f"Synthesis complete. Audio saved to: {result.export_audio_path}")

        if play_audio_after and result.export_audio_path:
            success = await self.player.play_async(result.export_audio_path)
            if success:
                logger.info(f"Playing audio: {result.export_audio_path}")
            else:
                logger.warning("Failed to play audio")

        return result

    def speak(
        self,
        text: str,
        wait_until_done: bool = False,
        voice_model: Optional[str] = None,
        **kwargs,
    ) -> bool:
        """
        Quick speak function - synthesizes and plays audio in one call.
        
        Args:
            text: Text to speak
            wait_until_done: Whether to wait until speech is complete
            voice_model: Voice model to use
            **kwargs: Additional RVC parameters
            
        Returns:
            True if successful, False otherwise
        """
        try:
            result = self.synthesize(
                text,
                voice_model=voice_model,
                play_audio_after=True,
                blocking_playback=wait_until_done,
                **kwargs,
            )
            return bool(result.export_audio_path)
        except Exception as e:
            logger.error(f"Error during speak: {e}")
            return False

    async def speak_async(
        self,
        text: str,
        voice_model: Optional[str] = None,
        **kwargs,
    ) -> bool:
        """
        Quick async speak function - synthesizes and plays audio.
        
        Args:
            text: Text to speak
            voice_model: Voice model to use
            **kwargs: Additional RVC parameters
            
        Returns:
            True if successful, False otherwise
        """
        try:
            result = await self.synthesize_async(
                text,
                voice_model=voice_model,
                play_audio_after=True,
                **kwargs,
            )
            return bool(result.export_audio_path)
        except Exception as e:
            logger.error(f"Error during speak_async: {e}")
            return False

    def stop_playback(self) -> None:
        """Stop current audio playback."""
        self.player.stop()

    def is_playing(self) -> bool:
        """Check if audio is currently playing."""
        return self.player.is_playing()
