from __future__ import annotations

import numpy as np

from .._metrics import confusion_matrix, f1_score, precision_score, recall_score
from ..domain.metrics import MetierMetrics


class MetierReporter:
    """Evaluates predictions with standard metrics and a business cost function.

    cost_fn: cost per false negative (missed positive — typically high).
    cost_fp: cost per false positive (false alarm — typically low).
    """

    def __init__(self, cost_fn: float = 10.0, cost_fp: float = 1.0) -> None:
        self._cost_fn = cost_fn
        self._cost_fp = cost_fp

    def report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str,
    ) -> MetierMetrics:
        _, fp, fn, _ = confusion_matrix(y_true, y_pred).ravel()
        return MetierMetrics(
            model_name=model_name,
            precision=float(precision_score(y_true, y_pred, zero_division=0)),
            recall=float(recall_score(y_true, y_pred, zero_division=0)),
            f1=float(f1_score(y_true, y_pred, zero_division=0)),
            business_cost=float(fn * self._cost_fn + fp * self._cost_fp),
        )

    def compare(self, results: list[MetierMetrics]) -> None:
        print("\n=== METIER COMPARISON ===")
        for m in sorted(results, key=lambda x: x.business_cost):
            print(m)
        best = min(results, key=lambda x: x.business_cost)
        print(f"\nChampion: {best.model_name}  (cost={best.business_cost:.0f})")
