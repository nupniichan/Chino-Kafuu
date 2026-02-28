import os
from dotenv import load_dotenv

load_dotenv()

# FastAPI settings
API_HOST: str = "127.0.0.1"
API_PORT: int = 8000
API_RELOAD: bool = True

# Project paths
PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STT_MODEL_PATH: str = os.path.join(PROJECT_ROOT, "models", "faster-whisper-small")
LLM_MODEL_PATH: str = os.path.join(PROJECT_ROOT, "models", "llm", "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf")

# Data paths
DATA_ROOT: str = os.path.join(PROJECT_ROOT, "data")
LONG_TERM_DB_PATH: str = os.path.join(DATA_ROOT, "memories", "conversations.db")

# OpenRouter settings
OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY") or ""
OPENROUTER_MODEL: str = "meta-llama/llama-3.3-70b-instruct:free"
OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
OPENROUTER_TIMEOUT: int = 60

# LLM settings
LLM_MODE: str = "openrouter"  # or "local"
LLM_N_CTX: int = 8192
LLM_N_GPU_LAYERS: int = 0
LLM_TEMPERATURE: float = 0.7
LLM_TOP_P: float = 0.9
LLM_MAX_TOKENS: int = 512

# Memory settings
SHORT_TERM_MEMORY_SIZE: int = 20
SHORT_TERM_TOKEN_LIMIT: int = 8192
MEMORY_IMPORTANCE_THRESHOLD: float = 0.8
IDLE_TIMEOUT_SECONDS: int = 30
MEMORY_CACHE: str = "in-memory"  # or "redis"

# Redis settings
REDIS_HOST: str = "localhost"
REDIS_PORT: int = 6379
REDIS_DB: int = 0

# VAD (Voice Activity Detection) settings
VAD_THRESHOLD: float = 0.5
SILENCE_CHUNKS_NEEDED: int = 5

# Audio settings
SAMPLE_RATE: int = 16000

# File upload limits
MAX_UPLOAD_SIZE_MB: int = 50
ALLOWED_AUDIO_FORMATS: tuple = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac")

# RVC (Voice Conversion) settings
RVC_BASE_URL: str = "http://127.0.0.1:6969/"