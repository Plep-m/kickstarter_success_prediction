from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class TextEncoderPort(Protocol):
    def fit_transform(self, texts: list[str]) -> np.ndarray: ...
    def transform(self, texts: list[str]) -> np.ndarray: ...
