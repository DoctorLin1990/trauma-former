"""
Training utilities: seed fixing, logging, device selection.
"""
import os
import random
import logging
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Fix all random seeds for exact reproducibility (Supplementary S2.4)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device(prefer_gpu: bool = True) -> torch.device:
    """Return the best available device."""
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Configure and return a named logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s  %(name)-25s  %(levelname)-8s  %(message)s",
                              datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
