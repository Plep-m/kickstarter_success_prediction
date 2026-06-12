from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ClassificationMetrics:
    """Accuracy and weighted-F1 on validation and test sets."""

    val_accuracy: float
    val_f1: float
    test_accuracy: float
    test_f1: float

    def __str__(self) -> str:
        return (
            f"val_acc={self.val_accuracy:.3f}  val_f1={self.val_f1:.3f}"
            f"  test_acc={self.test_accuracy:.3f}  test_f1={self.test_f1:.3f}"
        )


@dataclass(frozen=True)
class RegressionMetrics:
    """R², MAE and RMSE for a regression model."""

    r2: float
    mae: float
    rmse: float

    def __str__(self) -> str:
        return f"R2={self.r2:.3f}  MAE={self.mae:.3f}  RMSE={self.rmse:.3f}"


@dataclass(frozen=True)
class ClusterStats:
    """Inertia and silhouette score for a given k."""

    k: int
    inertia: float
    silhouette: float

    def __str__(self) -> str:
        return f"k={self.k}  inertia={self.inertia:>10.1f}  silhouette={self.silhouette:.3f}"


@dataclass(frozen=True)
class BootstrapResult:
    """Out-of-bag scores from all bootstrap iterations."""

    scores: tuple[float, ...]
    mean: float
    std: float

    def __str__(self) -> str:
        return f"Bootstrap({len(self.scores)} iters)  mean={self.mean:.3f}  std={self.std:.3f}"


@dataclass(frozen=True)
class CrossValResult:
    """K-fold cross-validation scores with aggregate statistics."""

    scores: tuple[float, ...]
    mean: float
    std: float
    k: int

    @property
    def is_stable(self) -> bool:
        return self.std < 0.02

    def __str__(self) -> str:
        fold_str = " ".join(f"{s:.3f}" for s in self.scores)
        label = "stable" if self.is_stable else "unstable"
        return (
            f"CrossVal(k={self.k})  [{fold_str}]\n"
            f"  mean={self.mean:.3f}  std={self.std:.3f}  → {label}"
        )


@dataclass(frozen=True)
class MetierMetrics:
    """Standard classification metrics plus a weighted business cost."""

    model_name: str
    precision: float
    recall: float
    f1: float
    business_cost: float

    def __str__(self) -> str:
        return (
            f"{self.model_name:<22} : "
            f"precision={self.precision:.2f}  recall={self.recall:.2f}  F1={self.f1:.2f}"
            f"  cost={self.business_cost:.0f}"
        )


@dataclass(frozen=True)
class FightResult:
    """Score and training time for one benchmark competitor."""

    name: str
    score: float
    train_time: float

    def __str__(self) -> str:
        return f"{self.name:<24} score={self.score:.4f}  ({self.train_time:.3f}s)"
