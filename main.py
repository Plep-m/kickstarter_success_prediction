from sklearn.model_selection import train_test_split

from mltoolbox._metrics import f1_score
from mltoolbox.adapters.sklearn import kickstarter_pipelines
from mltoolbox.data import KickstarterLoader, KickstarterCleaner
from mltoolbox.services.benchmark import Benchmark


def main() -> None:
    df = KickstarterLoader("data").load()
    X, y = KickstarterCleaner().build(df)
    print(f"Kickstarter: {len(X):,} rows  |  {y.mean():.1%} success rate")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    bench = (
        Benchmark(
            X_train, X_test, y_train, y_test,
            metric_fn=lambda yt, yp: f1_score(yt, yp, average="weighted"),
        )
        .register_many(kickstarter_pipelines())
        .run()
    )
    print(bench)


if __name__ == "__main__":
    main()
