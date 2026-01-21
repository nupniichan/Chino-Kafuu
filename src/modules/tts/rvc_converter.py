import logging
import os
import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

from gradio_client import Client

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RvcEnforceTerms2Request:
    terms_accepted: bool = False
    input_text_file_path: str = ""
    text_to_synthesize: str = ""
    tts_voice: str = "ja-JP-NanamiNeural"
    tts_speed: float = 0.0
    pitch: float = 0.0
    search_feature_ratio: float = 0.75
    volume_envelope: float = 1.0
    protect_voiceless_consonants: float = 0.5
    pitch_extraction_algorithm: str = "rmvpe"
    output_path_tts_audio: str = r"C:\Users\nup\Downloads\Applio\assets\audios\tts_output.wav"
    output_path_rvc_audio: str = r"C:\Users\nup\Downloads\Applio\assets\audios\tts_rvc_output.wav"
    voice_model: str = "Chino-Kafuu"
    index_file: Optional[str] = None
    split_audio: bool = False
    autotune: bool = False
    autotune_strength: float = 1.0
    proposed_pitch: bool = False
    proposed_pitch_threshold: float = 155.0
    clean_audio: bool = False
    clean_strength: float = 0.5
    export_format: str = "WAV"
    embedder_model: str = "contentvec"
    custom_embedder: Optional[str] = None
    speaker_id: Union[int, str] = 0

    voice_model_pth_path: Optional[str] = None
    voice_model_index_path: Optional[str] = None
    download_model_link: Optional[str] = None
    download_api_name: str = "/run_download_script"
    auto_download_if_missing: bool = True


@dataclass(frozen=True, slots=True)
class RvcEnforceTerms2Result:
    output_information: str
    export_audio_path: str


class RvcConverter:
    def __init__(self, base_url: str = "http://127.0.0.1:6969/") -> None:
        self._base_url = base_url
        self._client: Optional[Client] = None

    def _get_client(self) -> Client:
        if self._client is None:
            self._client = Client(self._base_url, download_files=False)
        return self._client

    @staticmethod
    def _ensure_parent_dir(file_path: str) -> None:
        if not file_path:
            return
        try:
            p = Path(file_path)
            if p.parent and not p.parent.exists():
                p.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            return

    @staticmethod
    def _normalize_path(path_value: Optional[str]) -> Optional[str]:
        if path_value is None:
            return None
        if not path_value:
            return ""
        return os.fspath(Path(path_value))

    @staticmethod
    def _files_exist(*paths: Optional[str]) -> bool:
        for p in paths:
            if not p:
                continue
            if not Path(p).exists():
                return False
        return True

    def _download_model_if_needed(self, req: RvcEnforceTerms2Request) -> Optional[str]:
        if not req.auto_download_if_missing:
            return None
        if not req.download_model_link:
            return None

        paths_to_check = [req.voice_model_pth_path, req.voice_model_index_path]
        has_any_path = any(bool(p) for p in paths_to_check)
        if not has_any_path:
            return None

        if self._files_exist(*paths_to_check):
            return None

        logger.info("Model files missing; calling Gradio API %s", req.download_api_name)
        output_info: str = self._get_client().predict(
            model_link=req.download_model_link,
            api_name=req.download_api_name,
        )

        if not self._files_exist(*paths_to_check):
            logger.warning(
                "Model download reported success but files are still missing locally. "
                "This can happen if the Gradio server stores models outside this project. "
                "pth=%r, index=%r. Output: %s",
                req.voice_model_pth_path,
                req.voice_model_index_path,
                output_info,
            )

        return str(output_info or "")

    def enforce_terms_2(self, req: RvcEnforceTerms2Request) -> RvcEnforceTerms2Result:
        if not req.text_to_synthesize:
            raise ValueError("text_to_synthesize is required")
        if not req.voice_model:
            raise ValueError("voice_model is required")

        self._ensure_parent_dir(req.output_path_tts_audio)
        self._ensure_parent_dir(req.output_path_rvc_audio)

        self._download_model_if_needed(req)

        index_value = req.index_file
        if not index_value and req.voice_model_index_path and Path(req.voice_model_index_path).exists():
            index_value = os.fspath(Path(req.voice_model_index_path))

        logger.info("Calling RVC Gradio API /enforce_terms_2")
        raw: Tuple[str, str] = self._get_client().predict(
            terms_accepted=req.terms_accepted,
            param_1=req.input_text_file_path,
            param_2=req.text_to_synthesize,
            param_3=req.tts_voice,
            param_4=req.tts_speed,
            param_5=req.pitch,
            param_6=req.search_feature_ratio,
            param_7=req.volume_envelope,
            param_8=req.protect_voiceless_consonants,
            param_9=req.pitch_extraction_algorithm,
            param_10=self._normalize_path(req.output_path_tts_audio) or "",
            param_11=self._normalize_path(req.output_path_rvc_audio) or "",
            param_12=req.voice_model,
            param_13=index_value,
            param_14=req.split_audio,
            param_15=req.autotune,
            param_16=req.autotune_strength,
            param_17=req.proposed_pitch,
            param_18=req.proposed_pitch_threshold,
            param_19=req.clean_audio,
            param_20=req.clean_strength,
            param_21=req.export_format,
            param_22=req.embedder_model,
            param_23=req.custom_embedder,
            param_24=req.speaker_id,
            api_name="/enforce_terms_2",
        )

        output_information, export_audio_path = raw
        return RvcEnforceTerms2Result(
            output_information=str(output_information or ""),
            export_audio_path=str(export_audio_path or ""),
        )

    async def enforce_terms_2_async(self, req: RvcEnforceTerms2Request) -> RvcEnforceTerms2Result:
        return await asyncio.to_thread(self.enforce_terms_2, req)
