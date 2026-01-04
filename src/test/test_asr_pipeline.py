import sounddevice as sd
import numpy as np
import logging
import queue
import sys
from pathlib import Path

# Add src directory to path for module imports
src_path = Path(__file__).parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from modules.asr.transcriber import Transcriber

# Configuration constants
SAMPLE_RATE = 16000  # Sample rate in Hz (must match VAD and STT requirements)
BLOCK_SIZE = 512  # Silero VAD requires exactly 512 samples for 16kHz (~32ms)
BLOCK_DURATION_MS = BLOCK_SIZE / SAMPLE_RATE * 1000  # Duration of each audio block in ms
SILENCE_CHUNKS_FOR_END_OF_SPEECH = 20  # Number of silent chunks before considering speech ended

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

def main():
    """
    Main function to run the ASR pipeline test.
    
    Workflow:
    1. Microphone -> sounddevice callback
    2. Callback converts audio (float32 -> int16 bytes) and puts into queue
    3. Main loop retrieves data from queue and feeds to Transcriber
    4. Transcriber performs VAD and STT, returns text if speech detected
    """
    logger = logging.getLogger("ASR_TEST")
    audio_queue = queue.Queue()

    def audio_callback(indata, frames, time, status):
        """Audio callback executed by sounddevice in a separate thread."""
        if status:
            logger.warning(f"Sounddevice status: {status}")
        
        # Convert audio from float32 to int16 bytes
        audio_int16 = (indata * 32767).astype(np.int16)
        audio_queue.put(bytearray(audio_int16.tobytes()))

    try:
        # Determine absolute path to model (from src/test/ -> project root -> models)
        project_root = Path(__file__).parent.parent.parent
        model_path = project_root / "models" / "faster-whisper-small"

        # Initialize Transcriber
        transcriber = Transcriber(
            vad_threshold=0.5,
            stt_model_path=str(model_path),
            silence_chunks_needed=SILENCE_CHUNKS_FOR_END_OF_SPEECH
        )

        # Open microphone audio stream
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            blocksize=BLOCK_SIZE,
            dtype='float32',
            callback=audio_callback
        )
        stream.start()
        logger.info("Microphone stream started. Speak now!")
        logger.info("Press Ctrl+C to stop")

        # Main processing loop
        while True:
            chunk = audio_queue.get()
            transcription = transcriber.process(chunk)
            if transcription:
                logger.info(f"[TRANSCRIPTION] {transcription}")

    except KeyboardInterrupt:
        logger.info("\nStopping test...")
    finally:
        if 'stream' in locals() and stream.active:
            stream.stop()
            stream.close()
        logger.info("Test finished.")

if __name__ == "__main__":
    main()