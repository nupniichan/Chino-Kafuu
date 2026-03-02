import logging
from typing import Optional

logger = logging.getLogger(__name__)

_CUDA_AVAILABLE: Optional[bool] = None


def is_cuda_available() -> bool:
    global _CUDA_AVAILABLE
    if _CUDA_AVAILABLE is None:
        try:
            import torch
            _CUDA_AVAILABLE = torch.cuda.is_available()
        except ImportError:
            _CUDA_AVAILABLE = False
        logger.info(f"CUDA available: {_CUDA_AVAILABLE}")
    return _CUDA_AVAILABLE


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if is_cuda_available() else "cpu"
    return device


def resolve_compute_type(compute_type: str, device: str) -> str:
    if compute_type == "auto":
        return "float16" if device == "cuda" else "int8"
    return compute_type


def resolve_gpu_layers(n_gpu_layers: int) -> int:
    if n_gpu_layers != 0 and not is_cuda_available():
        logger.warning(
            f"CUDA not available, overriding n_gpu_layers from {n_gpu_layers} to 0"
        )
        return 0
    return n_gpu_layers
