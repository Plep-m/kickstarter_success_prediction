"""Kickstarter Success Predictor — Streamlit UI.

Run with:
    streamlit run app.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src import data, models

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Kickstarter Predictor",
    page_icon="🚀",
    layout="wide",
)

# ── data helpers (cached) ─────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading dataset…")
def load_data() -> tuple[pd.DataFrame, pd.Series]:
    return data.load_clean("data")


@st.cache_data(show_spinner=False)
def get_options(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    return sorted(X["category"].unique()), sorted(X["country"].unique())


@st.cache_resource(show_spinner="Training model…")
def train_model(model_name: str):
    X, y = load_data()
    X_train, X_test, y_train, y_test = models.split(X, y)
    pipeline = models.build_models()[model_name]
    pipeline.fit(X_train, y_train)
    return pipeline, models.evaluate(pipeline, X_test, y_test)


# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Model")
    model_name = st.selectbox(
        "Classifier",
        ["xgboost", "random_forest", "logistic_regression"],
        format_func=lambda s: s.replace("_", " ").title(),
    )

    try:
        pipeline, test_metrics = train_model(model_name)
        st.success("Model ready")
        c1, c2 = st.columns(2)
        c1.metric("Accuracy",  f"{test_metrics['accuracy']:.3f}")
        c2.metric("AUC-ROC",   f"{test_metrics['roc_auc']:.3f}")
        c1.metric("Precision", f"{test_metrics['precision']:.3f}")
        c2.metric("Recall",    f"{test_metrics['recall']:.3f}")
        st.metric("F1 (succès)", f"{test_metrics['f1']:.3f}")
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    st.divider()
    st.caption(
        "Dataset: [Kickstarter Projects](https://www.kaggle.com/datasets/kemical/kickstarter-projects) · "
        "Features: category, goal, duration, country, description length, launch date"
    )

# ── main ─────────────────────────────────────────────────────────────────────

st.title("🚀 Kickstarter Success Predictor")
st.markdown(
    "Fill in your project details below and see how likely it is to be funded."
)

X_all, _ = load_data()
categories, countries = get_options(X_all)

# ── input form ────────────────────────────────────────────────────────────────

with st.form("predict_form"):
    col1, col2 = st.columns(2)

    with col1:
        category = st.selectbox("Category", categories)
        goal_usd = st.number_input(
            "Funding goal (USD)", min_value=100, max_value=1_000_000,
            value=10_000, step=500,
        )
        duration_days = st.slider("Campaign duration (days)", 1, 60, 30)

    with col2:
        country = st.selectbox("Country", countries)
        description_length = st.slider(
            "Description length (characters)", 10, 1000, 150,
            help="Approximate length of your project description / blurb."
        )
        launch_month = st.slider("Launch month", 1, 12, 6,
                                 format="%d",
                                 help="1 = January … 12 = December")
        launch_year = st.number_input(
            "Launch year", min_value=2009, max_value=2030, value=2025,
        )

    submitted = st.form_submit_button("Predict", use_container_width=True, type="primary")

# ── prediction ────────────────────────────────────────────────────────────────

if submitted:
    input_df = pd.DataFrame([{
        "category": category,
        "goal_usd": float(goal_usd),
        "duration_days": int(duration_days),
        "country": country,
        "description_length": int(description_length),
        "launch_year": int(launch_year),
        "launch_month": int(launch_month),
    }])

    proba = pipeline.predict_proba(input_df)[0]
    pred = int(pipeline.predict(input_df)[0])
    success_prob = float(proba[1])

    st.divider()
    res_col, gauge_col = st.columns([2, 1])

    with res_col:
        if pred == 1:
            st.success(f"## ✅ Likely to succeed  —  {success_prob:.1%} confidence")
        else:
            st.error(f"## ❌ Likely to fail  —  {success_prob:.1%} success chance")

    with gauge_col:
        st.metric("Success probability", f"{success_prob:.1%}")
        st.progress(success_prob)

    # ── sensitivity analysis ─────────────────────────────────────────────────
    st.divider()
    st.subheader("What if…")
    st.caption(
        "Each chart sweeps one parameter across its full range while keeping everything "
        "else fixed. The orange dot is your project. The dashed line is the 50% decision boundary."
    )
    sens_tab1, sens_tab2 = st.tabs(["Goal amount", "Campaign duration"])

    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    def _sweep(col: str, values: np.ndarray) -> list[float]:
        probas = []
        for v in values:
            row = input_df.copy()
            row[col] = v
            probas.append(float(pipeline.predict_proba(row)[0][1]))
        return probas

    def _sensitivity_chart(
        x_values: np.ndarray,
        probas: list[float],
        current_x: float,
        current_p: float,
        x_label: str,
        x_fmt: str = "{:.0f}",
    ) -> None:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(x_values, probas, color="steelblue", lw=2)
        ax.axhline(0.5, color="crimson", ls="--", lw=1, alpha=0.7, label="50% threshold")
        ax.axvline(current_x, color="darkorange", ls=":", lw=1.5, alpha=0.8)
        ax.scatter([current_x], [current_p], color="darkorange", s=80, zorder=5,
                   label=f"Your project  ({current_p:.0%})")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Success probability")
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.legend(fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with sens_tab1:
        goals = np.linspace(100, min(float(goal_usd) * 5, 500_000), 80)
        _sensitivity_chart(
            goals, _sweep("goal_usd", goals),
            float(goal_usd), success_prob,
            x_label="Funding goal (USD)",
        )

    with sens_tab2:
        durations = np.arange(1, 61, dtype=float)
        _sensitivity_chart(
            durations, _sweep("duration_days", durations),
            float(duration_days), success_prob,
            x_label="Campaign duration (days)",
        )
