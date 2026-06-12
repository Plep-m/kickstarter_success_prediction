from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BootstrapResult:
    scores: tuple[float, ...]
    mean: float
    std: float

    def __str__(self) -> str:
        return f"Bootstrap({len(self.scores)} iters)  mean={self.mean:.3f}  std={self.std:.3f}"


@dataclass(frozen=True)
class CrossValResult:
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
class FightResult:
    name: str
    score: float
    train_time: float

    def __str__(self) -> str:
        return f"{self.name:<24} score={self.score:.4f}  ({self.train_time:.3f}s)"
