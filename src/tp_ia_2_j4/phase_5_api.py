from __future__ import annotations

from pathlib import Path
import numpy as np

try:
    import joblib as _joblib
    _JOBLIB_AVAILABLE = True
except ImportError:
    _JOBLIB_AVAILABLE = False

try:
    from flask import Flask, jsonify, request
    _FLASK_AVAILABLE = True
except ImportError:
    _FLASK_AVAILABLE = False


class ModelSerializer:
    """Persists a (model, scaler) bundle with joblib to prevent normalisation mismatch."""

    def __init__(self, path: str = "model.joblib") -> None:
        if not _JOBLIB_AVAILABLE:
            raise ImportError("joblib required: pip install joblib")
        self._path = Path(path)

    def save(self, model: object, scaler: object) -> None:
        _joblib.dump({"model": model, "scaler": scaler}, self._path)
        print(f"Modèle sauvegardé → {self._path}")

    def load(self) -> tuple[object, object]:
        """Return (model, scaler) from disk."""
        bundle = _joblib.load(self._path)
        return bundle["model"], bundle["scaler"]


class PredictionAPI:
    """Minimal Flask REST API with a /predict POST endpoint."""

    def __init__(
        self,
        model_path: str,
        n_features: int,
        labels: list[str] | None = None,
    ) -> None:
        if not _FLASK_AVAILABLE:
            raise ImportError("flask required: pip install flask")
        self._serializer = ModelSerializer(model_path)
        self._n_features = n_features
        self._labels = labels or ["maligne", "benigne"]
        self._app: Flask | None = None

    def _build_app(self) -> Flask:
        model, scaler = self._serializer.load()
        n = self._n_features
        labels = self._labels

        app = Flask(__name__)

        @app.route("/predict", methods=["POST"])
        def predict():
            data = request.get_json(silent=True)
            if not data or "features" not in data:
                return jsonify({"error": "JSON avec clé 'features' requis"}), 400
            features = data["features"]
            if not isinstance(features, list) or len(features) != n:
                return jsonify({"error": f"Attendu {n} features, reçu {len(features) if isinstance(features, list) else '?'}"}), 400
            try:
                x = np.array(features, dtype=float).reshape(1, -1)
            except (ValueError, TypeError):
                return jsonify({"error": "Les features doivent être des nombres"}), 400
            x_scaled = scaler.transform(x)
            pred = int(model.predict(x_scaled)[0])
            proba = float(model.predict_proba(x_scaled)[0][pred])
            return jsonify({"prediction": pred, "proba": round(proba, 2), "label": labels[pred]})

        return app

    @property
    def app(self) -> Flask:
        if self._app is None:
            self._app = self._build_app()
        return self._app

    def run(self, host: str = "0.0.0.0", port: int = 5000, debug: bool = False) -> None:
        print(f"API démarrée sur http://{host}:{port}/predict")
        self.app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X, y = load_breast_cancer(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_tr, y_tr)

    ser = ModelSerializer("model.joblib")
    ser.save(model, scaler)

    api = PredictionAPI("model.joblib", n_features=X.shape[1])
    api.run(debug=True)
