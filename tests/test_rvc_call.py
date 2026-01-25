import asyncio
import logging
import sys
from pathlib import Path

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from modules.tts.rvc_converter import RvcConverter, RvcEnforceTerms2Request


async def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    converter = RvcConverter(base_url="http://127.0.0.1:6969/")
    req = RvcEnforceTerms2Request(
        terms_accepted=True,
        input_text_file_path="",
        text_to_synthesize="Hello!!",
        tts_voice="ja-JP-NanamiNeural",
        tts_speed=0,
        pitch=0,
        search_feature_ratio=0.75,
        volume_envelope=1,
        protect_voiceless_consonants=0.5,
        pitch_extraction_algorithm="rmvpe",
        output_path_tts_audio=r"C:\Users\nup\Downloads\Applio\assets\audios\tts_output.wav",
        output_path_rvc_audio=r"C:\Users\nup\Downloads\Applio\assets\audios\tts_rvc_output.wav",
        voice_model="Chino-Kafuu",
        index_file="",
        split_audio=False,
        autotune=False,
        autotune_strength=1,
        proposed_pitch=False,
        proposed_pitch_threshold=155,
        clean_audio=False,
        clean_strength=0.5,
        export_format="WAV",
        embedder_model="contentvec",
        custom_embedder=None,
        speaker_id=0,
    )

    result = await converter.enforce_terms_2_async(req)
    print(result.output_information)
    print(result.export_audio_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
