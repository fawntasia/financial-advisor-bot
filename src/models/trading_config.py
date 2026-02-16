"""Shared trading signal configuration and helpers."""

from typing import Mapping, Optional

import numpy as np
import pandas as pd

# Unified semantics across validation and backtesting:
# 1 -> long exposure, 0 -> flat exposure.
CLASS_TO_SIGNAL = {1: 1, 0: 0}


def predictions_to_signals(
    predictions,
    index=None,
    class_to_signal: Optional[Mapping[int, int]] = None,
) -> pd.Series:
    """Map class predictions to trading signals."""
    mapping = class_to_signal or CLASS_TO_SIGNAL
    pred_array = np.asarray(predictions).reshape(-1)
    mapped = [mapping.get(int(label), 0) for label in pred_array]
    return pd.Series(mapped, index=index, dtype=float)
