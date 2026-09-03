"""Device resolution helpers for training/inference."""

from __future__ import annotations

from vla_eval.core.logging import get_logger

logger = get_logger(__name__)


def resolve_device(requested: str = "cuda") -> str:
    """Resolve the best available device, gracefully degrading if unavailable.

    Args:
        requested: One of "cuda", "mps", or "cpu".

    Returns:
        The resolved device string, falling back to "cpu" if the requested
        accelerator is not available.
    """
    try:
        import torch
    except ImportError:
        return "cpu"

    if requested == "cuda" and not torch.cuda.is_available():
        logger.warning("cuda_unavailable_falling_back_to_cpu")
        return "cpu"
    if requested == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        logger.warning("mps_unavailable_falling_back_to_cpu")
        return "cpu"
    return requested
