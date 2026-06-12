from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer as _Tfidf


class TfidfAdapter:
    """Stateful TF-IDF wrapper — fit on train, transform test."""

    def __init__(self, **kwargs) -> None:
        self._vec = _Tfidf(**kwargs)
        self._fitted = False

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        self._fitted = True
        return self._vec.fit_transform(texts)

    def transform(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit_transform() on training texts first.")
        return self._vec.transform(texts)

    @property
    def vocabulary_size(self) -> int:
        return len(self._vec.vocabulary_)
