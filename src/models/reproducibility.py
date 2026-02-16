"""Reproducibility helpers for model training scripts."""

import os
import random
from typing import Optional

import numpy as np


def set_global_seeds(seed: int = 42, include_tensorflow: bool = False) -> Optional[str]:
    """
    Set common random seeds for reproducible runs.

    Returns:
        Optional status string for TensorFlow seeding path.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    if not include_tensorflow:
        return None

    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
        return "tensorflow_seeded"
    except Exception as exc:  # pragma: no cover - optional dependency path
        return f"tensorflow_seed_failed: {exc}"
