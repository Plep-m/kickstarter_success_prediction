"""Visualisation layer — the figures from the two CRISP-DM notebooks.

EDA charts (Phase 2 / notebook 02) and evaluation charts (Phase 5): each function
takes data and returns a matplotlib Figure, so it works both in a notebook
(`fig` displays inline) and in a script (`fig.savefig(...)`).
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

from .data import NUMERIC

sns.set_theme(style="whitegrid", palette="muted")

_LABELS = ["échec", "succès"]

def target_balance(y: pd.Series) -> plt.Figure:
    """Success vs failure proportion — shows the class imbalance."""
    rate = y.map({0: "échec", 1: "succès"}).value_counts(normalize=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    rate.plot(kind="bar", ax=ax, color=["#e74c3c", "#2ecc71"])
    ax.set_title("Répartition succès / échec")
    ax.set_ylabel("Proportion")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    for i, v in enumerate(rate.values):
        ax.text(i, v + 0.01, f"{v:.1%}", ha="center")
    fig.tight_layout()
    return fig


def success_by_category(X: pd.DataFrame, y: pd.Series) -> plt.Figure:
    """Success rate per category, vs the global mean."""
    rate = y.groupby(X["category"]).mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    rate.plot(kind="barh", ax=ax, color="steelblue")
    ax.axvline(y.mean(), color="red", ls="--", label="Moyenne globale")
    ax.set_title("Taux de succès par catégorie")
    ax.set_xlabel("Taux de succès")
    ax.legend()
    fig.tight_layout()
    return fig

def success_by_country(X: pd.DataFrame, y: pd.Series, top_n: int = 10) -> plt.Figure:
    """Success rate for the most frequent countries."""
    top = X["country"].value_counts().head(top_n).index
    mask = X["country"].isin(top)
    rate = y[mask].groupby(X.loc[mask, "country"]).mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    rate.plot(kind="bar", ax=ax, color="coral")
    ax.set_title(f"Taux de succès — top {top_n} pays")
    ax.set_ylabel("Taux de succès")
    ax.set_xlabel("Pays")
    plt.setp(ax.get_xticklabels(), rotation=45)
    fig.tight_layout()
    return fig

def goal_vs_duration(X: pd.DataFrame, y: pd.Series) -> plt.Figure:
    """Boxplots of goal and duration split by outcome (failed campaigns aim higher)."""
    df = X.assign(success=y.values)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.boxplot(data=df, x="success", y="goal_usd", ax=axes[0], showfliers=False)
    axes[0].set(title="Objectif USD vs succès", xlabel="Succès (0/1)", ylabel="goal_usd")
    sns.boxplot(data=df, x="success", y="duration_days", ax=axes[1], showfliers=False)
    axes[1].set(title="Durée (jours) vs succès", xlabel="Succès (0/1)")
    fig.tight_layout()
    return fig

def correlation(X: pd.DataFrame, y: pd.Series) -> plt.Figure:
    """Correlation heatmap of the numeric features and the target."""
    num = X[NUMERIC].assign(success=y.values)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(num.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, linewidths=0.5, ax=ax)
    ax.set_title("Matrice de corrélation")
    fig.tight_layout()
    return fig

def confusion_matrices(fitted: dict, X_test, y_test) -> plt.Figure:
    """Side-by-side confusion matrices for every fitted model."""
    fig, axes = plt.subplots(1, len(fitted), figsize=(5 * len(fitted), 4))
    axes = axes if len(fitted) > 1 else [axes]
    for ax, (name, model) in zip(axes, fitted.items()):
        ConfusionMatrixDisplay.from_estimator(
            model, X_test, y_test, ax=ax, colorbar=False, display_labels=_LABELS,
        )
        ax.set_title(name)
    fig.suptitle("Matrices de confusion — jeu test", y=1.02)
    fig.tight_layout()
    return fig


def roc_curves(fitted: dict, X_test, y_test) -> plt.Figure:
    """Overlaid ROC curves with the random-guess diagonal."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, model in fitted.items():
        RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, name=name)
    ax.plot([0, 1], [0, 1], "k--", label="Hasard")
    ax.set_title("Courbes ROC — jeu test")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def feature_importance(pipeline, top_n: int = 15) -> plt.Figure:
    """Bar chart of a tree model's top feature importances."""
    from .models import feature_importances

    imp = feature_importances(pipeline, top_n=top_n)
    fig, ax = plt.subplots(figsize=(8, 5))
    imp.sort_values().plot(kind="barh", ax=ax, color="seagreen")
    ax.set_title(f"Top {top_n} importances")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    return fig
