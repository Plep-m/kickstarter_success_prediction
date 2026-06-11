from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field

try:
    from tensorflow import keras
    from tensorflow.keras import layers as keras_layers
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


class KerasPMC:
    """Sequential PMC wrapper for binary classification using Keras."""

    def __init__(self, config: PMCConfig, n_features: int) -> None:
        if not _KERAS_AVAILABLE:
            raise ImportError("tensorflow is required: pip install tensorflow")
        self._config = config
        self._n_features = n_features
        self._model: keras.Sequential | None = None
        self._history: dict[str, list[float]] = {}

    def build(self) -> "KerasPMC":
        """Assemble the Sequential model from config."""
        layer_list: list = [keras_layers.Input(shape=(self._n_features,))]
        for units in self._config.hidden_units:
            layer_list.append(
                keras_layers.Dense(units, activation=self._config.hidden_activation)
            )
        layer_list.append(keras_layers.Dense(1, activation=self._config.output_activation))
        self._model = keras.Sequential(layer_list)
        self._model.compile(
            optimizer=self._config.optimizer,
            loss=self._config.loss,
            metrics=["accuracy"],
        )
        return self

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "KerasPMC":
        if self._model is None:
            self.build()
        val_data = (X_val, y_val) if X_val is not None else None
        hist = self._model.fit(
            X_train, y_train,
            epochs=self._config.epochs,
            batch_size=self._config.batch_size,
            validation_data=val_data,
            verbose=0,
        )
        self._history = hist.history
        return self

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
        """Return accuracy and loss on the test set."""
        if self._model is None:
            raise RuntimeError("Call fit() first.")
        loss, acc = self._model.evaluate(X_test, y_test, verbose=0)
        final_train_loss = float(self._history.get("loss", [float("nan")])[-1])
        print(f"Accuracy sur le test : {acc:.2%}")
        print(f"Loss finale (train)  : {final_train_loss:.4f}")
        return {"accuracy": float(acc), "loss": float(loss), "train_loss_final": final_train_loss}

    @property
    def model(self) -> "keras.Sequential":
        if self._model is None:
            raise RuntimeError("Call build() or fit() first.")
        return self._model

    @property
    def loss_curve(self) -> list[float]:
        """Training loss per epoch."""
        return self._history.get("loss", [])


if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X, y = load_breast_cancer(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    pmc = KerasPMC(PMCConfig(), n_features=X_tr.shape[1])
    pmc.fit(X_tr, y_tr)
    pmc.evaluate(X_te, y_te)
