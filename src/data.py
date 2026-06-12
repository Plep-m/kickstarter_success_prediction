"""Business layer — the Kickstarter domain.

What counts as a project, how we define "success", and which features describe it.
No machine learning here, only data rules.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# The columns the models will see.
NUMERIC = ["goal_usd", "duration_days", "description_length", "launch_year", "launch_month"]
CATEGORICAL = ["category", "country"]
FEATURES = CATEGORICAL + NUMERIC

_CSV_NAMES = ["ks-projects-201801.csv", "ks-projects-201612.csv"]


def load(data_dir: str = "data") -> pd.DataFrame:
    """Find a Kickstarter CSV in `data_dir`, read it, normalise column names."""
    folder = Path(data_dir)
    path = next((folder / n for n in _CSV_NAMES if (folder / n).exists()), None)
    if path is None:
        matches = sorted(folder.glob("ks-*.csv")) if folder.exists() else []
        if not matches:
            raise FileNotFoundError(
                f"No Kickstarter CSV in '{folder}'. "
                "Download ks-projects-201801.csv from Kaggle and place it in data/."
            )
        path = matches[0]

    try:
        df = pd.read_csv(path, encoding="utf-8", low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin1", low_memory=False)

    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)
    return df


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Keep successful/failed projects, engineer features, return (X, y)."""
    data = df[df["state"].isin(["successful", "failed"])].copy()

    deadline = pd.to_datetime(data["deadline"], errors="coerce")
    launched = pd.to_datetime(data["launched"], errors="coerce")
    data["duration_days"] = (deadline - launched).dt.days
    data["launch_year"] = launched.dt.year
    data["launch_month"] = launched.dt.month

    text_col = next((c for c in ["blurb", "name"] if c in data.columns), "name")
    data["description_length"] = data[text_col].astype(str).str.len()

    data["goal_usd"] = pd.to_numeric(
        data["usd_goal_real"] if "usd_goal_real" in data.columns else data["goal"],
        errors="coerce",
    )
    data["category"] = data["main_category"] if "main_category" in data.columns else data["category"]

    y = (data["state"] == "successful").astype(int)
    X = data[FEATURES]

    valid = X.notna().all(axis=1) & (data["duration_days"] > 0) & (data["goal_usd"] > 0)
    return X[valid].reset_index(drop=True), y[valid].reset_index(drop=True)


def load_clean(data_dir: str = "data") -> tuple[pd.DataFrame, pd.Series]:
    """Convenience: load + clean in one call. The single source both entry points use."""
    return clean(load(data_dir))
