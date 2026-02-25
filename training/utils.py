"""
Utility functions for training: random seed fixing, logging setup.
"""
import random
import numpy as np
import torch
import logging
import os
import sys
from typing import Optional

def set_seed(seed: int = 42, deterministic: bool = True):
    """
    Set random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed: integer seed.
        deterministic: if True, sets CuDNN to deterministic mode (may slow down training).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def setup_logger(name: str, log_file: Optional[str] = None, level=logging.INFO) -> logging.Logger:
    """
    Set up logger with console and optional file output.

    Args:
        name: logger name.
        log_file: path to log file (if None, no file output).
        level: logging level.

    Returns:
        Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler (optional)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)