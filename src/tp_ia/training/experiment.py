from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, f1_score

from tp_ia.data.dataset import Dataset


class Experiment:
    """Trains and evaluates a set of models against a Dataset's splits."""

    def __init__(self, dataset: Dataset, models: dict[str, BaseEstimator]):
        self.dataset = dataset
        self.models = models
        self._scores: dict[str, dict[str, float]] = {}

    def run(self) -> "Experiment":
        for name, model in self.models.items():
            model.fit(self.dataset.training.features, self.dataset.training.labels)

            val_predictions = model.predict(self.dataset.validation.features)
            test_predictions = model.predict(self.dataset.testing.features)

            self._scores[name] = {
                "val_accuracy": accuracy_score(self.dataset.validation.labels, val_predictions),
                "val_f1": f1_score(self.dataset.validation.labels, val_predictions, average="weighted"),
                "test_accuracy": accuracy_score(self.dataset.testing.labels, test_predictions),
                "test_f1": f1_score(self.dataset.testing.labels, test_predictions, average="weighted"),
            }
        return self

    @property
    def scores(self) -> dict[str, dict[str, float]]:
        if not self._scores:
            raise RuntimeError("Call run() before accessing scores.")
        return self._scores

    def __str__(self) -> str:
        if not self._scores:
            return "No results yet — call run() first."
        col = 24
        header = f"{'Model':<{col}} {'Val Acc':>8} {'Val F1':>8} {'Test Acc':>9} {'Test F1':>8}"
        sep = "-" * len(header)
        rows = [header, sep]
        for name, metrics in self._scores.items():
            rows.append(
                f"{name:<{col}} {metrics['val_accuracy']:>8.3f} {metrics['val_f1']:>8.3f}"
                f" {metrics['test_accuracy']:>9.3f} {metrics['test_f1']:>8.3f}"
            )
        return "\n".join(rows)
