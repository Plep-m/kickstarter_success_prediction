from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


@dataclass(frozen=True)
class MetierMetrics:
    """Standard classification metrics plus a weighted business cost."""

    model_name: str
    precision: float
    recall: float
    f1: float
    business_cost: float

    def __str__(self) -> str:
        return (
            f"{self.model_name:<22} : "
            f"precision={self.precision:.2f}  recall={self.recall:.2f}  F1={self.f1:.2f}"
            f" | coût métier = {self.business_cost:.0f}"
        )


class MetierReporter:
    """Evaluates predictions with both standard metrics and a business cost function.

    cout_fn: cost per false negative (missed positive — typically high).
    cout_fp: cost per false positive (false alarm — typically low).
    """

    def __init__(self, cout_fn: float = 10.0, cout_fp: float = 1.0) -> None:
        self._cout_fn = cout_fn
        self._cout_fp = cout_fp

    def report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str,
    ) -> MetierMetrics:
        """Compute metrics and business cost for one model."""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return MetierMetrics(
            model_name=model_name,
            precision=float(precision_score(y_true, y_pred, zero_division=0)),
            recall=float(recall_score(y_true, y_pred, zero_division=0)),
            f1=float(f1_score(y_true, y_pred, zero_division=0)),
            business_cost=float(fn * self._cout_fn + fp * self._cout_fp),
        )

    def compare(self, results: list[MetierMetrics]) -> None:
        """Print all results sorted by ascending business cost."""
        print("\n=== COMPARAISON MÉTIER ===")
        for m in sorted(results, key=lambda x: x.business_cost):
            print(m)
        best = min(results, key=lambda x: x.business_cost)
        print(f"\nChampion métier : {best.model_name} (coût = {best.business_cost:.0f})")


if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X, y = load_breast_cancer(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    sc = StandardScaler()
    X_tr, X_te = sc.fit_transform(X_tr), sc.transform(X_te)

    reporter = MetierReporter(cout_fn=10, cout_fp=1)
    results: list[MetierMetrics] = []
    for name, clf in [
        ("RandomForest", RandomForestClassifier(n_estimators=100, random_state=42)),
        ("GradientBoosting", GradientBoostingClassifier(random_state=42)),
    ]:
        clf.fit(X_tr, y_tr)
        results.append(reporter.report(y_te, clf.predict(X_te), name))
    reporter.compare(results)
