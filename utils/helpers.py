"""
Miscellaneous utilities: seed fixing, device selection, logging helpers.
"""

import os
import random
import numpy as np
import torch


def fix_seed(seed: int = 1001):
    """Fix random seeds for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(preferred: str = "auto") -> torch.device:
    """
    Return the best available device.

    Args:
        preferred: 'auto' (detect), 'cuda', 'mps', or 'cpu'.

    Returns:
        torch.device
    """
    if preferred != "auto":
        return torch.device(preferred)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model: torch.nn.Module) -> int:
    """Return the total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(state: dict, path: str):
    """Save a training checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, device: torch.device) -> dict:
    """Load a training checkpoint onto the given device."""
    return torch.load(path, map_location=device)
