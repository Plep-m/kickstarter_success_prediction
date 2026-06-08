from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.utils import Bunch

DatasetName = Literal["iris", "wine", "breast_cancer", "digits"]

@dataclass(frozen=True)
class DataChunk:
    """Container for a specific chunk of data."""
    features: np.ndarray
    labels: np.ndarray

    @property
    def dimensions(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Return shapes of (features, labels) as a convenience."""
        return self.features.shape, self.labels.shape


class Dataset:
    """Loads a dataset, filters it, and splits into training/validation/test sets."""

    _DATASET_LOADERS: dict[str, Callable[[], Bunch]] = {
        "iris": datasets.load_iris,
        "wine": datasets.load_wine,
        "breast_cancer": datasets.load_breast_cancer,
        "digits": datasets.load_digits,
    }

    def __init__(
        self, 
        name: DatasetName, 
        keep_only_class: int | None = None,
        test_fraction: float = 0.15,
        validation_fraction: float = 0.15,
        random_seed: int | None = 42
    ):
        if name not in self._DATASET_LOADERS:
            raise ValueError(
                f"Unknown dataset '{name}'. Available: {list(self._DATASET_LOADERS)}"
            )

        self.name: str = name
        self._raw_data: Bunch = self._DATASET_LOADERS[name]()
        
        self._all_features: np.ndarray = self._raw_data.data
        self._all_labels: np.ndarray = self._raw_data.target

        if keep_only_class is not None:
            keep_mask = self._all_labels == keep_only_class
            self._all_features = self._all_features[keep_mask]
            self._all_labels = self._all_labels[keep_mask]

        self._create_splits(test_fraction, validation_fraction, random_seed)

    def _create_splits(
        self, test_fraction: float, validation_fraction: float, random_seed: int | None
    ) -> None:
        use_balanced_split = self._all_labels if len(np.unique(self._all_labels)) > 1 else None

        temp_features, test_features, temp_labels, test_labels = train_test_split(
            self._all_features, self._all_labels, 
            test_size=test_fraction, 
            random_state=random_seed, 
            stratify=use_balanced_split
        )

        use_balanced_split_temp = temp_labels if use_balanced_split is not None else None
        adjusted_validation_fraction = validation_fraction / (1.0 - test_fraction)

        train_features, validation_features, train_labels, validation_labels = train_test_split(
            temp_features, temp_labels, 
            test_size=adjusted_validation_fraction, 
            random_state=random_seed, 
            stratify=use_balanced_split_temp
        )

        self._training = DataChunk(train_features, train_labels)
        self._validation = DataChunk(validation_features, validation_labels)
        self._testing = DataChunk(test_features, test_labels)

    @property
    def all_features(self) -> np.ndarray:
        """Return the complete feature matrix."""
        return self._all_features

    @property
    def all_labels(self) -> np.ndarray:
        """Return the complete target vector."""
        return self._all_labels

    @property
    def training(self) -> DataChunk:
        """Return the training subset."""
        return self._training

    @property
    def validation(self) -> DataChunk:
        """Return the validation subset."""
        return self._validation

    @property
    def testing(self) -> DataChunk:
        """Return the test subset."""
        return self._testing

    @classmethod
    def list_available_datasets(cls) -> list[str]:
        """List all available dataset names."""
        return list(cls._DATASET_LOADERS)

    def __str__(self) -> str:
        """Return a formatted summary of the dataset shapes and classes with percentages."""
        total_rows, total_columns = self.all_features.shape
        output = [f"Rows, columns: ({total_rows}, {total_columns})"]

        class_labels, counts = np.unique(self.all_labels, return_counts=True)
        total_cases = len(self.all_labels)
        target_names = getattr(self._raw_data, "target_names", class_labels)

        for label, count in zip(class_labels, counts):
            class_name = target_names[label]
            percentage = (count / total_cases) * 100 if total_cases > 0 else 0
            output.append(
                f"Class {label} ({class_name}): {count} cases ({percentage:.1f}%)"
            )

        return "\n".join(output)
