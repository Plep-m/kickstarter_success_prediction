from __future__ import annotations

from typing import Iterator, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class KFoldSplitterPort(Protocol):
    def splits(self, X: np.ndarray, y: np.ndarray) -> Iterator[tuple[np.ndarray, np.ndarray]]: ...

    @property
    def k(self) -> int: ...
