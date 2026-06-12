from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier as _RFC
from sklearn.feature_selection import mutual_info_classif as _mi_classif


class RFFeatureSelectorAdapter:
    """Ranks features by Random Forest importance and mutual information."""

    def __init__(self, n_estimators: int = 100, random_state: int = 42) -> None:
        self._rf = _RFC(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)

    def rank(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        corr = X.corrwith(y).abs().rename("corr_target")
        mi = pd.Series(
            _mi_classif(X, y, random_state=42),
            index=X.columns,
            name="mutual_info",
        )
        self._rf.fit(X, y)
        rf_imp = pd.Series(
            self._rf.feature_importances_,
            index=X.columns,
            name="rf_importance",
        )
        return pd.DataFrame({"corr_target": corr, "mutual_info": mi, "rf_importance": rf_imp})
