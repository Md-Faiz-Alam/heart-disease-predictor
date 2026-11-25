"""Reusable widgets and charts."""

import threading
from io import BytesIO

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import shap
import streamlit as st

from heart.config import CATEGORY_LABELS, FEATURE_LABELS, NUMERIC_RANGES
from heart.predict import HIGH_RISK_CUTOFF, Prediction

GREEN, AMBER, RED = "#2E7D32", "#F9A825", "#C62828"
BAND_COLORS = {"Low": GREEN, "Moderate": AMBER, "High": RED}


def category_input(container, name: str, default: int, key: str | None = None) -> int:
    options = list(CATEGORY_LABELS[name])
    return container.selectbox(
        FEATURE_LABELS[name],
        options,
        index=options.index(int(default)),
        format_func=lambda code, n=name: CATEGORY_LABELS[n][code],
        key=key,
    )


def number_input(container, name: str, default: float, step: float = 1.0) -> float:
    low, high = NUMERIC_RANGES[name]
    cast = float if step < 1 else int
    return container.number_input(
        FEATURE_LABELS[name],
        min_value=cast(low),
        max_value=cast(high),
        value=cast(default),
        step=cast(step),
    )


def risk_gauge(probability: float, threshold: float) -> go.Figure:
    high = max(HIGH_RISK_CUTOFF, threshold) * 100
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=round(probability * 100, 1),
            number={"suffix": "%", "font": {"size": 44}},
            gauge={
                "axis": {"range": [0, 100], "ticksuffix": "%"},
                "bar": {"color": "rgba(30,30,30,0.85)", "thickness": 0.25},
                "steps": [
                    {"range": [0, threshold * 100], "color": "rgba(46,125,50,0.35)"},
                    {"range": [threshold * 100, high], "color": "rgba(249,168,37,0.40)"},
                    {"range": [high, 100], "color": "rgba(198,40,40,0.40)"},
                ],
                "threshold": {
                    "line": {"color": RED, "width": 3},
                    "thickness": 0.9,
                    "value": threshold * 100,
                },
            },
        )
    )
    fig.update_layout(height=230, margin={"t": 20, "b": 0, "l": 30, "r": 30})
    return fig


# shap.plots draws on pyplot's global "current figure", which is shared by every Streamlit
# session thread. Without this lock, overlapping reruns corrupt each other's figures.
_PYPLOT_LOCK = threading.Lock()


def waterfall_png(prediction: Prediction, max_display: int = 10) -> bytes:
    contribs = prediction.contributions
    explanation = shap.Explanation(
        values=np.array([c.shap_value for c in contribs]),
        base_values=prediction.base_probability,
        data=np.array([c.display_value for c in contribs], dtype=object),
        feature_names=[c.label for c in contribs],
    )
    with _PYPLOT_LOCK:
        fig = plt.figure(figsize=(9, 5.5))
        try:
            shap.plots.waterfall(explanation, max_display=max_display, show=False)
            fig = plt.gcf()
            fig.set_size_inches(9, 5.5)
            fig.tight_layout()
            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
        finally:
            plt.close(fig)
    return buf.getvalue()


def factor_lists(prediction: Prediction, limit: int = 3) -> None:
    raises = [c for c in prediction.contributions if c.shap_value > 0.01][:limit]
    lowers = [c for c in prediction.contributions if c.shap_value < -0.01][:limit]
    if raises:
        st.markdown("**Raising the estimate**")
        for c in raises:
            st.markdown(f"- {c.label}: *{c.display_value}* (+{c.shap_value:.0%})")
    if lowers:
        st.markdown("**Lowering the estimate**")
        for c in lowers:
            st.markdown(f"- {c.label}: *{c.display_value}* ({c.shap_value:.0%})")
