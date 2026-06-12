"""Kickstarter predictor.

- `data`   — business layer: load + clean, define "success".
- `models` — math layer: preprocessing, models, cross-validation, scoring.
- `plots`  — visualisation: EDA and evaluation figures from the notebooks.
"""
from . import data, models, plots

__all__ = ["data", "models", "plots"]
