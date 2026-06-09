import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif


def features_discriminantes(df: pd.DataFrame, cible: str = "Churn") -> pd.DataFrame:
    """Rank features by predictive power using correlation and Random Forest.

    Returns a DataFrame with both scores indexed by feature name.
    """
    X = df.drop(columns=[cible])
    y = df[cible]

    corr_cible = X.corrwith(y).abs().sort_values(ascending=False)

    mi = mutual_info_classif(X, y, random_state=42)
    mi_serie = pd.Series(mi, index=X.columns).sort_values(ascending=False)

    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    rf_importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

    classement = pd.DataFrame({
        "corr_cible": corr_cible,
        "mutual_info": mi_serie,
        "rf_importance": rf_importances,
    })

    print("Top 10 — Corrélation à la cible :")
    print(corr_cible.head(10).to_string())
    print("\nTop 10 — Information mutuelle :")
    print(mi_serie.head(10).to_string())
    print("\nTop 10 — Importance Random Forest :")
    print(rf_importances.head(10).to_string())

    return classement
