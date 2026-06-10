from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


_NUMERIC_COLS: list[str] = [
    "price",
    "minimum_nights",
    "number_of_reviews",
    "availability_365",
    "reviews_per_month",
]


@dataclass(frozen=True)
class ClusterStats:
    """Inertia and silhouette score for one value of k."""

    k: int
    inertia: float
    silhouette: float

    def __str__(self) -> str:
        return f"k={self.k} : inertie={self.inertia:>9.1f}  silhouette={self.silhouette:.3f}"


class AirbnbLoader:
    """Loads an Airbnb listings CSV, retains numeric columns, drops NaN rows.

    Provide an Inside Airbnb listings.csv.gz URL or a local path.
    """

    def __init__(
        self,
        url_or_path: str,
        numeric_cols: list[str] = _NUMERIC_COLS,
    ) -> None:
        self._source = url_or_path
        self._cols = numeric_cols

    def load(self) -> pd.DataFrame:
        """Return a clean numeric-only DataFrame."""
        df = pd.read_csv(
            self._source,
            usecols=lambda c: c in self._cols,
            low_memory=False,
        )
        df = df[[c for c in self._cols if c in df.columns]].dropna()
        print(
            f"Listings chargés : {len(df)} lignes, "
            f"{df.shape[1]} colonnes numériques retenues"
        )
        return df


class KOptimizer:
    """Sweeps k values and tracks inertia + silhouette to locate the elbow."""

    def __init__(
        self,
        k_range: range = range(2, 9),
        n_init: int = 10,
        random_state: int = 42,
    ) -> None:
        self._k_range = k_range
        self._n_init = n_init
        self._random_state = random_state
        self._stats: list[ClusterStats] = []

    def fit(self, X_scaled: np.ndarray) -> "KOptimizer":
        """Run KMeans for every k and collect ClusterStats."""
        for k in self._k_range:
            km = KMeans(
                n_clusters=k,
                n_init=self._n_init,
                random_state=self._random_state,
            ).fit(X_scaled)
            sil = float(silhouette_score(X_scaled, km.labels_))
            self._stats.append(
                ClusterStats(k=k, inertia=float(km.inertia_), silhouette=sil)
            )
        return self

    @property
    def stats(self) -> list[ClusterStats]:
        if not self._stats:
            raise RuntimeError("Call fit() first.")
        return self._stats

    def best_k(self) -> int:
        """Return k with highest silhouette score."""
        return max(self._stats, key=lambda s: s.silhouette).k


class AirbnbSegmenter:
    """Full pipeline: load → standardise → sweep k → report segments."""

    def __init__(
        self,
        url_or_path: str,
        k_range: range = range(2, 9),
    ) -> None:
        self._loader = AirbnbLoader(url_or_path)
        self._k_range = k_range
        self._optimizer: KOptimizer | None = None
        self._scaler = StandardScaler()

    def run(self) -> "AirbnbSegmenter":
        df = self._loader.load()
        X_scaled = self._scaler.fit_transform(df.values)
        self._optimizer = KOptimizer(self._k_range).fit(X_scaled)
        return self

    def report(self) -> None:
        if self._optimizer is None:
            raise RuntimeError("Call run() first.")
        for stat in self._optimizer.stats:
            print(stat)
        print(f"\nSegment retenu : k={self._optimizer.best_k()}")

    @property
    def optimizer(self) -> KOptimizer:
        if self._optimizer is None:
            raise RuntimeError("Call run() first.")
        return self._optimizer


if __name__ == "__main__":
    import sys

    source = sys.argv[1] if len(sys.argv) > 1 else "listings.csv"
    AirbnbSegmenter(source).run().report()
