from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import joblib as _joblib
    _JOBLIB_AVAILABLE = True
except ImportError:
    _JOBLIB_AVAILABLE = False


class JoblibModelStore:
    """Persists a (model, scaler) bundle to disk using joblib."""

    def __init__(self, path: str = "model.joblib") -> None:
        if not _JOBLIB_AVAILABLE:
            raise ImportError("joblib is required: pip install joblib")
        self._path = Path(path)

    def save(self, model: Any, scaler: Any) -> None:
        _joblib.dump({"model": model, "scaler": scaler}, self._path)

    def load(self) -> tuple[Any, Any]:
        bundle = _joblib.load(self._path)
        return bundle["model"], bundle["scaler"]

    @property
    def path(self) -> Path:
        return self._path
