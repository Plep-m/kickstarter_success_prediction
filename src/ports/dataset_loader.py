from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class DatasetLoaderPort(Protocol):
    def load(self) -> tuple[np.ndarray, np.ndarray]: ...

    @property
    def feature_names(self) -> list[str]: ...
