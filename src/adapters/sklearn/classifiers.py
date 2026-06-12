from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier as _GBC
from sklearn.ensemble import RandomForestClassifier as _RFC
from sklearn.linear_model import LogisticRegression as _LR
from sklearn.linear_model import SGDClassifier as _SGD
from sklearn.naive_bayes import GaussianNB as _GNB
from sklearn.naive_bayes import MultinomialNB as _MNB
from sklearn.pipeline import make_pipeline as _pipeline
from sklearn.preprocessing import StandardScaler as _SS
from sklearn.svm import SVC as _SVC
from sklearn.tree import DecisionTreeClassifier as _DT

try:
    from xgboost import XGBClassifier as _XGB
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False


class _BaseClassifierAdapter:
    """Mixin: delegates fit/predict/predict_proba to self._model."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_BaseClassifierAdapter":
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._model!r})"


class LogisticRegressionAdapter(_BaseClassifierAdapter):
    def __init__(self, max_iter: int = 1000, random_state: int = 42, scale: bool = True) -> None:
        lr = _LR(max_iter=max_iter, random_state=random_state)
        self._model = _pipeline(_SS(), lr) if scale else lr


class SGDClassifierAdapter(_BaseClassifierAdapter):
    def __init__(self, max_iter: int = 1000, random_state: int = 42, scale: bool = True) -> None:
        sgd = _SGD(max_iter=max_iter, random_state=random_state)
        self._model = _pipeline(_SS(), sgd) if scale else sgd


class RandomForestClassifierAdapter(_BaseClassifierAdapter):
    def __init__(self, n_estimators: int = 100, random_state: int = 42, n_jobs: int = -1) -> None:
        self._model = _RFC(n_estimators=n_estimators, random_state=random_state, n_jobs=n_jobs)


class GradientBoostingClassifierAdapter(_BaseClassifierAdapter):
    def __init__(self, random_state: int = 42) -> None:
        self._model = _GBC(random_state=random_state)


class SVCAdapter(_BaseClassifierAdapter):
    def __init__(
        self,
        kernel: str = "rbf",
        random_state: int = 42,
        scale: bool = True,
        probability: bool = True,
    ) -> None:
        svc = _SVC(kernel=kernel, random_state=random_state, probability=probability)
        self._model = _pipeline(_SS(), svc) if scale else svc


class DecisionTreeClassifierAdapter(_BaseClassifierAdapter):
    def __init__(self, random_state: int = 42) -> None:
        self._model = _DT(random_state=random_state)


class GaussianNBAdapter(_BaseClassifierAdapter):
    def __init__(self) -> None:
        self._model = _GNB()


class MultinomialNBAdapter(_BaseClassifierAdapter):
    def __init__(self) -> None:
        self._model = _MNB()


class XGBoostClassifierAdapter(_BaseClassifierAdapter):
    def __init__(self, random_state: int = 42) -> None:
        if not _XGB_AVAILABLE:
            raise ImportError("xgboost is required: pip install xgboost")
        self._model = _XGB(random_state=random_state, eval_metric="logloss", verbosity=0)
