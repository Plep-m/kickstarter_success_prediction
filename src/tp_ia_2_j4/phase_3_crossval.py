from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold, cross_val_score


@dataclass(frozen=True)
class CrossValResult:
    """K-fold cross-validation scores with aggregate statistics."""

    scores: tuple[float, ...]
    mean: float
    std: float
    k: int

    def __str__(self) -> str:
        scores_str = " ".join(f"{s:.3f}" for s in self.scores)
        stability = "stable" if self.std < 0.02 else "instable — grand écart-type, vérifier les données"
        return (
            f"Scores par fold : [{scores_str}]\n"
            f"Moyenne : {self.mean:.3f} | Écart-type : {self.std:.3f}  →  modèle {stability}"
        )


class CrossValidator:
    """Stratified k-fold cross-validation with stability reporting."""

    def __init__(
        self,
        k: int = 5,
        scoring: str = "accuracy",
        random_state: int = 42,
    ) -> None:
        self._k = k
        self._scoring = scoring
        self._cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)

    def evaluate(self, model: BaseEstimator, X: np.ndarray, y: np.ndarray) -> CrossValResult:
        """Run k-fold CV and return aggregated CrossValResult."""
        scores = cross_val_score(model, X, y, cv=self._cv, scoring=self._scoring)
        return CrossValResult(
            scores=tuple(float(s) for s in scores),
            mean=float(scores.mean()),
            std=float(scores.std()),
            k=self._k,
        )


if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer
    from sklearn.ensemble import RandomForestClassifier

    X, y = load_breast_cancer(return_X_y=True)
    result = CrossValidator(k=5).evaluate(RandomForestClassifier(n_estimators=100, random_state=42), X, y)
    print(result)
