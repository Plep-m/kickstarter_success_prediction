from mltoolbox.adapters.sklearn import (
    SklearnDatasetAdapter,
    StratifiedSplitterAdapter,
    StandardScalerAdapter,
    LogisticRegressionAdapter,
    RandomForestClassifierAdapter,
    StratifiedKFoldAdapter,
)
from mltoolbox.services import Experiment, CrossValidator, BootstrapEvaluator, MetierReporter

def main() -> None:
    X, y = SklearnDatasetAdapter("breast_cancer").load()

    split = StratifiedSplitterAdapter().split(X, y)
    scaler = StandardScalerAdapter()
    split.train = split.train.__class__(scaler.fit_transform(split.train.features), split.train.labels)
    split.val   = split.val.__class__(scaler.transform(split.val.features),   split.val.labels)
    split.test  = split.test.__class__(scaler.transform(split.test.features), split.test.labels)

    models = {
        "logistic_regression": LogisticRegressionAdapter(scale=False),
        "random_forest":       RandomForestClassifierAdapter(n_estimators=100),
    }

    print("=== Experiment ===")
    print(Experiment(split, models).run())

    print("\n=== Cross-validation ===")
    print(CrossValidator(StratifiedKFoldAdapter(k=5)).evaluate(LogisticRegressionAdapter(), X, y))

    print("\n=== Bootstrap ===")
    print(BootstrapEvaluator(RandomForestClassifierAdapter(n_estimators=50), n_iterations=20).evaluate(X, y))

    print("\n=== Metier ===")
    lr = LogisticRegressionAdapter()
    lr.fit(split.train.features, split.train.labels)
    reporter = MetierReporter(cost_fn=10, cost_fp=1)
    print(reporter.report(split.test.labels, lr.predict(split.test.features), "logistic"))


if __name__ == "__main__":
    main()
