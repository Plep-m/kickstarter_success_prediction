"""Benchmark the three models on the Kickstarter dataset.

Run with: python main.py
"""
from src import data, models


def main() -> None:
    X, y = data.load_clean()
    print(f"Kickstarter: {len(X):,} rows  |  {y.mean():.1%} success rate\n")

    X_train, X_test, y_train, y_test = models.split(X, y)

    print("Leaderboard (weighted F1):")
    for name, f1 in models.benchmark(X_train, X_test, y_train, y_test):
        print(f"  {name:<22} {f1:.4f}")


if __name__ == "__main__":
    main()
