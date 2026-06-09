import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def split_et_scale_proprement(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split then scale: scaler is fit on train only, applied (transform) to test."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)  # transform only — no fit on test

    return X_train_scaled, X_test_scaled, np.asarray(y_train), np.asarray(y_test)


def bilan_nettoyage(metriques: dict[str, tuple]) -> None:
    """Print a before/after cleaning summary table.

    metriques: {"Étape": (avant, apres), ...}
    """
    print(f"\n{'Étape':<30} {'Avant':>8} {'Après':>8}")
    print("-" * 48)
    for etape, (avant, apres) in metriques.items():
        print(f"{etape:<30} {str(avant):>8} {str(apres):>8}")


def explorer_pca(
    X_scaled: np.ndarray,
    y: np.ndarray,
    variance_cible: float = 0.90,
) -> None:
    """Project to 2D with PCA, plot colored by target, report explained variance."""
    pca_full = PCA().fit(X_scaled)
    variance_cumulee = np.cumsum(pca_full.explained_variance_ratio_)
    n_pour_cible = int(np.argmax(variance_cumulee >= variance_cible) + 1)

    print(f"Variance expliquée avec 2 composantes : {variance_cumulee[1]:.1%}")
    print(f"Composantes pour {variance_cible:.0%} de variance : {n_pour_cible}")

    X_2d = PCA(n_components=2).fit_transform(X_scaled)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap="coolwarm", alpha=0.5, s=10)
    plt.xlabel("Composante principale 1")
    plt.ylabel("Composante principale 2")
    plt.title("PCA — projection 2D colorée par Churn")
    plt.colorbar(scatter, label="0 = Non, 1 = Oui")
    plt.tight_layout()
    plt.show()


def comparer_strategies(
    X_honnete: np.ndarray,
    X_triche: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """Train a logistic regression on both splits and compare accuracy/F1.

    X_honnete: test set scaled with scaler fit on train only.
    X_triche:  test set scaled with scaler fit on full X before split.
    """
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_honnete[:len(y_train)], y_train)

    for label, X_test in [("Honnête (train-only scaler)", X_honnete), ("Triche (scaler sur tout X)", X_triche)]:
        preds = model.predict(X_test[len(y_train):] if X_test.shape[0] > len(y_test) else X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="weighted")
        print(f"{label:<40} Accuracy={acc:.4f}  F1={f1:.4f}")
