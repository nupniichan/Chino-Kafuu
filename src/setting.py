"""
Application configuration settings.
Contains paths, audio parameters, and API server settings.
"""
import os

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
MID_TERM_DB_PATH: str = os.path.join(DATA_ROOT, "memories", "conversations.db")
LONG_TERM_DB_PATH: str = os.path.join(DATA_ROOT, "memories", "vector_db")

# LLM settings
LLM_N_CTX: int = 8192  # Context window size
LLM_N_GPU_LAYERS: int = 0  # GPU layers (0 = CPU only, increase for GPU)
LLM_TEMPERATURE: float = 0.7  # Response creativity (0.0 - 1.0)
LLM_TOP_P: float = 0.9  # Nucleus sampling
LLM_MAX_TOKENS: int = 512  # Maximum response length

# Memory settings
SHORT_TERM_MEMORY_SIZE: int = 20  # Maximum messages in conversation buffer
SHORT_TERM_TOKEN_LIMIT: int = 8192  # Token limit before compression
MEMORY_IMPORTANCE_THRESHOLD: float = 0.8  # Score threshold for long-term storage
IDLE_TIMEOUT_SECONDS: int = 30  # Idle time before auto-trigger event

# Redis settings
REDIS_HOST: str = "localhost"
REDIS_PORT: int = 6379
REDIS_DB: int = 0

# VAD (Voice Activity Detection) settings
VAD_THRESHOLD: float = 0.5  # Speech probability threshold (0.0 - 1.0)
SILENCE_CHUNKS_NEEDED: int = 5  # Number of silent chunks before considering speech ended

# Audio settings
SAMPLE_RATE: int = 16000  # Audio sample rate in Hz (required by Silero VAD)
CHUNK_SIZE: int = 512  # Audio chunk size in samples (required by Silero VAD for 16kHz)

# File upload limits
MAX_UPLOAD_SIZE_MB: int = 50  # Maximum upload file size in megabytes
ALLOWED_AUDIO_FORMATS: tuple = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac")

