from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

DEFAULT_MODELS: dict[str, BaseEstimator] = {
    "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "sgd": make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, random_state=42)),
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "gradient_boosting": GradientBoostingClassifier(random_state=42),
    "xgboost": XGBClassifier(random_state=42, eval_metric="logloss", verbosity=0),
}
