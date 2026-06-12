from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:
    from tensorflow import keras as _keras
    from tensorflow.keras import layers as _layers
    _KERAS_AVAILABLE = True
except ImportError:
    _KERAS_AVAILABLE = False


@dataclass
class PMCConfig:
    """Hyperparameters for a Sequential Keras binary classifier."""

    hidden_units: list[int] = field(default_factory=lambda: [16, 8])
    hidden_activation: str = "relu"
    output_activation: str = "sigmoid"
    loss: str = "binary_crossentropy"
    optimizer: str = "adam"
    epochs: int = 20
    batch_size: int = 16


class KerasPMCAdapter:
    """Sequential multi-layer perceptron for binary classification, adapting ClassifierPort."""

    def __init__(self, config: PMCConfig, n_features: int) -> None:
        if not _KERAS_AVAILABLE:
            raise ImportError("tensorflow is required: pip install tensorflow")
        self._config = config
        self._n_features = n_features
        self._model = None
        self._history: dict[str, list[float]] = {}

    def _build(self) -> None:
        layers = [_layers.Input(shape=(self._n_features,))]
        for units in self._config.hidden_units:
            layers.append(_layers.Dense(units, activation=self._config.hidden_activation))
        layers.append(_layers.Dense(1, activation=self._config.output_activation))
        self._model = _keras.Sequential(layers)
        self._model.compile(
            optimizer=self._config.optimizer,
            loss=self._config.loss,
            metrics=["accuracy"],
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "KerasPMCAdapter":
        if self._model is None:
            self._build()
        val_data = (X_val, y_val) if X_val is not None else None
        hist = self._model.fit(
            X, y,
            epochs=self._config.epochs,
            batch_size=self._config.batch_size,
            validation_data=val_data,
            verbose=0,
        )
        self._history = hist.history
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call fit() first.")
        return (self._model.predict(X, verbose=0).squeeze() >= 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call fit() first.")
        p = self._model.predict(X, verbose=0).squeeze()
        return np.column_stack([1 - p, p])

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        if self._model is None:
            raise RuntimeError("Call fit() first.")
        loss, acc = self._model.evaluate(X, y, verbose=0)
        return {
            "accuracy": float(acc),
            "loss": float(loss),
            "train_loss_final": float(self._history.get("loss", [float("nan")])[-1]),
        }

    @property
    def loss_curve(self) -> list[float]:
        return self._history.get("loss", [])
