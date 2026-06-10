from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


@dataclass(frozen=True)
class SpamMetrics:
    """Precision / recall / F1 on the spam class."""

    precision: float
    recall: float
    f1: float

    def __str__(self) -> str:
        return f"precision={self.precision:.3f}  recall={self.recall:.3f}  f1={self.f1:.3f}"


class SpamLoader:
    """Loads the SMS Spam Collection (UCI tab-separated: label\\tmessage)."""

    def __init__(self, path: str) -> None:
        self._path = path

    def load(self) -> tuple[list[str], np.ndarray]:
        """Return (texts, binary_labels) where 1 = spam."""
        df = pd.read_csv(self._path, sep="\t", header=None, names=["label", "text"])
        texts: list[str] = df["text"].tolist()
        labels: np.ndarray = (df["label"] == "spam").astype(int).values
        print(f"SMS chargés : {len(texts)} messages, {labels.sum()} spams ({labels.mean():.1%})")
        return texts, labels


class SpamEncoder:
    """Stateful TF-IDF wrapper — fit on train, transform test."""

    def __init__(self) -> None:
        self._vec = TfidfVectorizer()
        self._fitted = False

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        self._fitted = True
        return self._vec.fit_transform(texts)

    def transform(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit_transform() on training data first.")
        return self._vec.transform(texts)


class SpamEvaluator:
    """Fits one classifier and reports spam-class P/R/F1."""

    def __init__(self, model: BaseEstimator, name: str) -> None:
        self._model = model
        self._name = name

    def fit_evaluate(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
    ) -> SpamMetrics:
        self._model.fit(X_train, y_train)
        y_pred = self._model.predict(X_test)
        print(f"\n--- {self._name} ---")
        print(classification_report(y_test, y_pred, target_names=["normal", "spam"], zero_division=0))
        p, r, f, _ = precision_recall_fscore_support(
            y_test, y_pred, pos_label=1, average="binary", zero_division=0
        )
        return SpamMetrics(precision=float(p), recall=float(r), f1=float(f))


class SpamBenchmark:
    """Trains NaiveBayes and LogisticRegression on shared TF-IDF features."""

    _MODELS: dict[str, BaseEstimator] = {
        "NaiveBayes": MultinomialNB(),
        "LogisticRegression": LogisticRegression(max_iter=5000, random_state=42),
    }

    def __init__(
        self,
        path: str,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> None:
        self._loader = SpamLoader(path)
        self._encoder = SpamEncoder()
        self._test_size = test_size
        self._random_state = random_state
        self._results: dict[str, SpamMetrics] = {}

    def run(self) -> "SpamBenchmark":
        texts, labels = self._loader.load()
        t_train, t_test, y_train, y_test = train_test_split(
            texts, labels,
            test_size=self._test_size,
            random_state=self._random_state,
            stratify=labels,
        )
        X_train = self._encoder.fit_transform(t_train)
        X_test = self._encoder.transform(t_test)
        for name, model in self._MODELS.items():
            self._results[name] = SpamEvaluator(model, name).fit_evaluate(
                X_train, X_test, y_train, y_test
            )
        return self

    def report(self) -> None:
        print("\n=== RÉSUMÉ SPAM ===")
        for name, metrics in self._results.items():
            print(f"{name:<22} : {metrics}")

    @property
    def results(self) -> dict[str, SpamMetrics]:
        if not self._results:
            raise RuntimeError("Call run() first.")
        return self._results


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "SMSSpamCollection"
    SpamBenchmark(path).run().report()
