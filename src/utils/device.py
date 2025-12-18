"""Utility functions for device management and reproducibility."""

import os
import random
from typing import Optional

import numpy as np
import torch


def get_device(device: Optional[str] = None) -> torch.device:
    """Get the best available device with fallback chain: CUDA -> MPS -> CPU.
    
    Args:
        device: Preferred device ('auto', 'cpu', 'cuda', 'mps')
        
    Returns:
        torch.device: The selected device
    """
    if device is None or device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # For MPS (Apple Silicon)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"


def setup_environment(seed: int = 42, deterministic: bool = True) -> torch.device:
    """Setup the environment for reproducible training.
    
    Args:
        seed: Random seed value
        deterministic: Whether to use deterministic algorithms
        
    Returns:
        torch.device: The selected device
    """
    device = get_device()
    
    if deterministic:
        set_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True
        
    print(f"Using device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if hasattr(torch.backends, "mps"):
        print(f"MPS available: {torch.backends.mps.is_available()}")
    
    return device
