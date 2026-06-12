from __future__ import annotations

import numpy as np

from ..ports.model_store import ModelStorePort

try:
    from flask import Flask, jsonify, request as _request
    _FLASK_AVAILABLE = True
except ImportError:
    _FLASK_AVAILABLE = False


class PredictionAPI:
    """Minimal Flask REST API that serves a stored (model, scaler) bundle.

    POST /predict — JSON body: {"features": [f1, f2, ...]}
    Returns: {"prediction": int, "proba": float, "label": str}
    """

    def __init__(
        self,
        store: ModelStorePort,
        n_features: int,
        labels: list[str] | None = None,
    ) -> None:
        if not _FLASK_AVAILABLE:
            raise ImportError("flask is required: pip install flask")
        self._store = store
        self._n_features = n_features
        self._labels = labels or ["class_0", "class_1"]
        self._app: Flask | None = None

    def _build(self) -> Flask:
        model, scaler = self._store.load()
        n = self._n_features
        labels = self._labels

        app = Flask(__name__)

        @app.route("/predict", methods=["POST"])
        def predict():
            data = _request.get_json(silent=True)
            if not data or "features" not in data:
                return jsonify({"error": "JSON body with 'features' key required"}), 400
            feats = data["features"]
            if not isinstance(feats, list) or len(feats) != n:
                got = len(feats) if isinstance(feats, list) else "?"
                return jsonify({"error": f"Expected {n} features, got {got}"}), 400
            try:
                x = np.array(feats, dtype=float).reshape(1, -1)
            except (ValueError, TypeError):
                return jsonify({"error": "All features must be numeric"}), 400
            x_scaled = scaler.transform(x)
            pred = int(model.predict(x_scaled)[0])
            proba = float(model.predict_proba(x_scaled)[0][pred])
            return jsonify({"prediction": pred, "proba": round(proba, 4), "label": labels[pred]})

        return app

    @property
    def app(self) -> Flask:
        if self._app is None:
            self._app = self._build()
        return self._app

    def run(self, host: str = "0.0.0.0", port: int = 5000, debug: bool = False) -> None:
        self.app.run(host=host, port=port, debug=debug)
