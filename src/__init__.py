"""ML/DL toolbox with hexagonal architecture.

Layer overview
--------------
domain/     Pure dataclasses — no framework imports.
ports/      Protocol definitions (structural interfaces for adapters).
adapters/   All sklearn / keras / joblib / flask imports live here.
services/   Orchestration using ports — framework-agnostic business logic.
data/       Pandas-based data loading and preprocessing.
neural/     Pure-numpy building blocks (Neuron, Activations, GradientDescent).
"""
from .domain.dataset import DataChunk, SplitResult
from .domain.metrics import (
    BootstrapResult,
    ClassificationMetrics,
    ClusterStats,
    CrossValResult,
    FightResult,
    MetierMetrics,
    RegressionMetrics,
)
from .services.benchmark import Benchmark
from .services.bootstrap import BootstrapEvaluator
from .services.cross_validation import CrossValidator
from .services.experiment import Experiment
from .services.metier import MetierReporter

__version__ = "0.1.0"

__all__ = [
    "DataChunk",
    "SplitResult",
    "BootstrapResult",
    "ClassificationMetrics",
    "ClusterStats",
    "CrossValResult",
    "FightResult",
    "MetierMetrics",
    "RegressionMetrics",
    "Benchmark",
    "BootstrapEvaluator",
    "CrossValidator",
    "Experiment",
    "MetierReporter",
]
