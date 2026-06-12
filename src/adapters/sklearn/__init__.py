"""Default sklearn adapter registries — ready-to-use instances for experiments."""
from .classifiers import (
    DecisionTreeClassifierAdapter,
    GaussianNBAdapter,
    GradientBoostingClassifierAdapter,
    LogisticRegressionAdapter,
    MultinomialNBAdapter,
    RandomForestClassifierAdapter,
    SGDClassifierAdapter,
    SVCAdapter,
)
from .clusterers import KMeansAdapter
from .datasets import SklearnDatasetAdapter
from .feature_selectors import RFFeatureSelectorAdapter
from .regressors import (
    LinearRegressionAdapter,
    RandomForestRegressorAdapter,
    RidgeRegressionAdapter,
)
from .scalers import (
    MinMaxScalerAdapter,
    RobustScalerAdapter,
    StandardScalerAdapter,
)
from .splitters import StratifiedKFoldAdapter, StratifiedSplitterAdapter
from .text_encoders import TfidfAdapter

DEFAULT_CLASSIFIERS: dict[str, object] = {
    "logistic_regression": LogisticRegressionAdapter(),
    "sgd": SGDClassifierAdapter(),
    "random_forest": RandomForestClassifierAdapter(),
    "gradient_boosting": GradientBoostingClassifierAdapter(),
    "decision_tree": DecisionTreeClassifierAdapter(),
    "svc_rbf": SVCAdapter(kernel="rbf"),
    "gaussian_nb": GaussianNBAdapter(),
}

DEFAULT_REGRESSORS: dict[str, object] = {
    "linear_regression": LinearRegressionAdapter(),
    "ridge": RidgeRegressionAdapter(),
    "random_forest": RandomForestRegressorAdapter(),
}

__all__ = [
    "DecisionTreeClassifierAdapter",
    "GaussianNBAdapter",
    "GradientBoostingClassifierAdapter",
    "KMeansAdapter",
    "LinearRegressionAdapter",
    "LogisticRegressionAdapter",
    "MinMaxScalerAdapter",
    "MultinomialNBAdapter",
    "RFFeatureSelectorAdapter",
    "RandomForestClassifierAdapter",
    "RandomForestRegressorAdapter",
    "RidgeRegressionAdapter",
    "RobustScalerAdapter",
    "SGDClassifierAdapter",
    "SVCAdapter",
    "SklearnDatasetAdapter",
    "StandardScalerAdapter",
    "StratifiedKFoldAdapter",
    "StratifiedSplitterAdapter",
    "TfidfAdapter",
    "DEFAULT_CLASSIFIERS",
    "DEFAULT_REGRESSORS",
]
