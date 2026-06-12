from __future__ import annotations

from .._metrics import accuracy_score, f1_score
from ..domain.dataset import SplitResult
from ..domain.metrics import ClassificationMetrics
from ..ports.classifier import ClassifierPort


class Experiment:
    """Trains and evaluates a dict of classifiers against a SplitResult."""

    def __init__(self, split: SplitResult, models: dict[str, ClassifierPort]) -> None:
        self._split = split
        self._models = models
        self._scores: dict[str, ClassificationMetrics] = {}

    def run(self) -> "Experiment":
        for name, model in self._models.items():
            model.fit(self._split.train.features, self._split.train.labels)
            val_pred = model.predict(self._split.val.features)
            test_pred = model.predict(self._split.test.features)
            self._scores[name] = ClassificationMetrics(
                val_accuracy=accuracy_score(self._split.val.labels, val_pred),
                val_f1=f1_score(self._split.val.labels, val_pred, average="weighted"),
                test_accuracy=accuracy_score(self._split.test.labels, test_pred),
                test_f1=f1_score(self._split.test.labels, test_pred, average="weighted"),
            )
        return self

    @property
    def scores(self) -> dict[str, ClassificationMetrics]:
        if not self._scores:
            raise RuntimeError("Call run() before accessing scores.")
        return self._scores

    def best(self, metric: str = "test_accuracy") -> str:
        return max(self._scores, key=lambda n: getattr(self._scores[n], metric))

    def __str__(self) -> str:
        if not self._scores:
            return "No results — call run() first."
        col = 26
        header = f"{'Model':<{col}} {'ValAcc':>8} {'ValF1':>8} {'TestAcc':>9} {'TestF1':>8}"
        sep = "-" * len(header)
        rows = [header, sep]
        for name, m in self._scores.items():
            rows.append(
                f"{name:<{col}} {m.val_accuracy:>8.3f} {m.val_f1:>8.3f}"
                f" {m.test_accuracy:>9.3f} {m.test_f1:>8.3f}"
            )
        return "\n".join(rows)
