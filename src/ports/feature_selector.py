from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd


@runtime_checkable
class FeatureSelectorPort(Protocol):
    """Ranks features by predictive power and returns a scored DataFrame."""

    def rank(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame: ...
