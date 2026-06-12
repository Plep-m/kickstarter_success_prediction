"""Full CRISP-DM run on the Kickstarter dataset, from the terminal.

Cross-validation → final test evaluation → classification report, and exports the
results table + evaluation figures to results/. Run with: python main.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: save figures without a display
from sklearn.metrics import classification_report

from src import data, models, plots

RESULTS = Path("results")
FIGURES = RESULTS / "figures"


def main() -> None:
    # ── Phases 2-3 : données ────────────────────────────────────────────────
    X, y = data.load_clean()
    print(f"Données : {len(X):,} projets  |  {y.mean():.1%} de succès")
    X_train, X_test, y_train, y_test = models.split(X, y)

    # ── Phase 4 : validation croisée (5-fold sur le train) ──────────────────
    print("\n=== Validation croisée (5-fold stratifié, jeu d'entraînement) ===")
    cv = models.cross_validate(X_train, y_train)
    print(cv[models.SCORING].round(3).to_string())

    # ── Phase 5 : évaluation finale sur le jeu test ─────────────────────────
    print("\n=== Évaluation finale (jeu test, classé par AUC) ===")
    test, fitted = models.benchmark(X_train, X_test, y_train, y_test)
    print(test.round(3).to_string())

    champion = test.index[0]
    print(f"\nChampion (AUC test) : {champion}\n")
    print(classification_report(
        y_test, fitted[champion].predict(X_test), target_names=["échec", "succès"]
    ))

    # ── Exports ─────────────────────────────────────────────────────────────
    FIGURES.mkdir(parents=True, exist_ok=True)
    cv.join(test, how="outer", lsuffix="_cv", rsuffix="_test").to_csv(
        RESULTS / "kickstarter_benchmark.csv"
    )
    plots.confusion_matrices(fitted, X_test, y_test).savefig(
        FIGURES / "confusion_matrices.png", bbox_inches="tight"
    )
    plots.roc_curves(fitted, X_test, y_test).savefig(
        FIGURES / "roc_curves.png", bbox_inches="tight"
    )
    plots.feature_importance(fitted["random_forest"]).savefig(
        FIGURES / "feature_importance.png", bbox_inches="tight"
    )
    print(f"Résultats et figures exportés dans {RESULTS}/")


if __name__ == "__main__":
    main()
