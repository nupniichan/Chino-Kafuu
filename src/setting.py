import os
from dotenv import load_dotenv

load_dotenv()

# FastAPI settings
API_HOST: str = os.getenv("API_HOST") or ""
API_PORT: int = int(os.getenv("API_PORT") or "0")
API_RELOAD: bool = (os.getenv("API_RELOAD") or "False").lower() == "true"

# Project paths
PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STT_MODEL_PATH: str = os.path.join(PROJECT_ROOT, "models", "faster-whisper-small")
LLM_MODEL_PATH: str = os.path.join(PROJECT_ROOT, "models", "llm", "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf")

# Data paths
DATA_ROOT: str = os.path.join(PROJECT_ROOT, "data")
LONG_TERM_DB_PATH: str = os.path.join(DATA_ROOT, "memories", "conversations.db")

# OpenRouter settings
OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY") or ""
OPENROUTER_MODEL: str = os.getenv("OPENROUTER_MODEL") or ""
OPENROUTER_BASE_URL: str = os.getenv("OPENROUTER_BASE_URL") or ""
OPENROUTER_TIMEOUT: int = int(os.getenv("OPENROUTER_TIMEOUT") or "0")

# LLM settings
LLM_MODE: str = os.getenv("LLM_MODE") or ""
LLM_N_CTX: int = int(os.getenv("LLM_N_CTX") or "0")
LLM_N_GPU_LAYERS: int = int(os.getenv("LLM_N_GPU_LAYERS") or "0")
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE") or "0")
LLM_TOP_P: float = float(os.getenv("LLM_TOP_P") or "0")
LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS") or "0")

# Memory settings
SHORT_TERM_MEMORY_SIZE: int = int(os.getenv("SHORT_TERM_MEMORY_SIZE") or "0")
SHORT_TERM_TOKEN_LIMIT: int = int(os.getenv("SHORT_TERM_TOKEN_LIMIT") or "0")
MEMORY_IMPORTANCE_THRESHOLD: float = float(os.getenv("MEMORY_IMPORTANCE_THRESHOLD") or "0")
IDLE_TIMEOUT_SECONDS: int = int(os.getenv("IDLE_TIMEOUT_SECONDS") or "0")
MEMORY_CACHE: str = os.getenv("MEMORY_CACHE") or ""

# Redis settings
REDIS_HOST: str = os.getenv("REDIS_HOST") or ""
REDIS_PORT: int = int(os.getenv("REDIS_PORT") or "0")
REDIS_DB: int = int(os.getenv("REDIS_DB") or "0")

# VAD (Voice Activity Detection) settings
VAD_THRESHOLD: float = float(os.getenv("VAD_THRESHOLD") or "0")
SILENCE_CHUNKS_NEEDED: int = int(os.getenv("SILENCE_CHUNKS_NEEDED") or "0")

# Audio settings
SAMPLE_RATE: int = int(os.getenv("SAMPLE_RATE") or "0")
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE") or "0")

# File upload limits
MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_SIZE_MB") or "0")
ALLOWED_AUDIO_FORMATS: tuple = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac")