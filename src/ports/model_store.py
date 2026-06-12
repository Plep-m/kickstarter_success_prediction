from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ModelStorePort(Protocol):
    def save(self, model: Any, scaler: Any) -> None: ...
    def load(self) -> tuple[Any, Any]: ...
