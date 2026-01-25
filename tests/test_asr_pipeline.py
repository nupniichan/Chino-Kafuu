import sounddevice as sd
import numpy as np
import logging
import queue
import sys
from pathlib import Path

src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from modules.asr.transcriber import Transcriber

SAMPLE_RATE = 16000
BLOCK_SIZE = 512
BLOCK_DURATION_MS = BLOCK_SIZE / SAMPLE_RATE * 1000
SILENCE_CHUNKS_FOR_END_OF_SPEECH = 20

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

def main():
    logger = logging.getLogger("ASR_TEST")
    audio_queue = queue.Queue()

    def audio_callback(indata, frames, time, status):
        if status:
            logger.warning(f"Sounddevice status: {status}")
        
        audio_int16 = (indata * 32767).astype(np.int16)
        audio_queue.put(bytearray(audio_int16.tobytes()))

    try:
        project_root = Path(__file__).parent.parent
        model_path = project_root / "models" / "faster-whisper-small"

        transcriber = Transcriber(
            vad_threshold=0.5,
            stt_model_path=str(model_path),
            silence_chunks_needed=SILENCE_CHUNKS_FOR_END_OF_SPEECH
        )

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
