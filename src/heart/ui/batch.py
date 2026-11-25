"""Batch screening tab: upload a CSV, validate every row, score, download results."""

import uuid

import pandas as pd
import plotly.express as px
import streamlit as st

from heart.config import FEATURES
from heart.observability import PredictionLogger
from heart.predict import HeartModel
from heart.schemas import EXAMPLE_PATIENT, parse_patients
from heart.ui.components import BAND_COLORS


def template_csv() -> bytes:
    rows = [
        {"patient_id": "P001", **EXAMPLE_PATIENT},
        {
            "patient_id": "P002",
            "age": 37,
            "sex": 1,
            "cp": 3,
            "trestbps": 130,
            "chol": 250,
            "fbs": 0,
            "restecg": 0,
            "thalach": 187,
            "exang": 0,
            "oldpeak": 3.5,
            "slope": 3,
            "ca": 0,
            "thal": 3,
        },
    ]
    return pd.DataFrame(rows, columns=["patient_id", *FEATURES]).to_csv(index=False).encode()


def batch_tab(model: HeartModel, prediction_log: PredictionLogger, max_rows: int) -> None:
    st.subheader("Batch screening")
    st.caption(
        "Upload a CSV with one patient per row, using the 13 UCI feature columns and "
        "original codes. "
        "Any extra columns (for example `patient_id`) are passed through to the results "
        "but never logged."
    )
    st.download_button(
        "Download CSV template", template_csv(), "heart_batch_template.csv", "text/csv"
    )

    upload = st.file_uploader("Patient CSV", type=["csv"])
    if upload is None:
        return
    try:
        df = pd.read_csv(upload)
    except Exception as exc:  # malformed upload: tell the user, never crash the app
        st.error(f"Could not read CSV: {exc}")
        return
    if len(df) > max_rows:
        st.error(f"Too many rows: {len(df)} (limit {max_rows}).")
        return
    try:
        valid, errors = parse_patients(df)
    except ValueError as exc:
        st.error(str(exc))
        return

    if errors:
        st.warning(f"{len(errors)} row(s) failed validation and were skipped.")
        st.dataframe(pd.DataFrame(errors), hide_index=True)
    if not valid:
        return

    index = [i for i, _ in valid]
    X = pd.concat([p.to_frame() for _, p in valid], ignore_index=True)
    X.index = index
    scored = model.predict_batch(X)

    request_id = uuid.uuid4().hex
    for (_, patient), row in zip(valid, scored.itertuples(), strict=True):
        prediction_log.log(
            request_id=request_id,
            source="ui-batch",
            model_version=model.version,
            inputs=patient.model_dump(),
            probability=float(row.probability),
            has_disease=bool(row.has_disease),
        )

    results = df.loc[index].join(scored)
    results["probability"] = results["probability"].round(4)
    results = results.sort_values("probability", ascending=False)

    c1, c2, c3 = st.columns(3)
    c1.metric("Patients scored", len(results))
    c2.metric("Screen-positive", int(results["has_disease"].sum()))
    c3.metric("Threshold", f"{model.threshold:.2f}")

    fig = px.histogram(
        results,
        x="probability",
        color="risk_band",
        nbins=20,
        range_x=[0, 1],
        color_discrete_map=BAND_COLORS,
        category_orders={"risk_band": list(BAND_COLORS)},
        labels={"probability": "Estimated probability", "risk_band": "Risk band"},
    )
    fig.add_vline(x=model.threshold, line_dash="dash", annotation_text="threshold")
    fig.update_layout(height=300, margin={"t": 30, "b": 10})
    st.plotly_chart(fig)

    st.dataframe(results, hide_index=True)
    st.download_button(
        "Download results CSV",
        results.to_csv(index=False).encode(),
        "heart_batch_results.csv",
        "text/csv",
        type="primary",
    )
