"""Streamlit UI for the heart disease risk model.

Run:  streamlit run app.py
"""

import json
import uuid
from dataclasses import dataclass
from datetime import datetime

import streamlit as st
from pydantic import ValidationError

from heart.config import FEATURE_LABELS, MODEL_CARD_PATH, NUMERIC_RANGES, REPORTS_DIR
from heart.observability import PredictionLogger
from heart.predict import HeartModel, Prediction
from heart.report import build_pdf_report
from heart.schemas import EXAMPLE_PATIENT, PatientInput
from heart.settings import get_settings
from heart.ui.batch import batch_tab
from heart.ui.components import (
    BAND_COLORS,
    category_input,
    factor_lists,
    number_input,
    risk_gauge,
    waterfall_png,
)
from heart.ui.performance import performance_tab, sidebar

st.set_page_config(page_title="Heart Disease Risk", page_icon="🫀", layout="wide")

# Real patients from the UCI Cleveland data (rows 2 and 4), in the original encodings.
EXAMPLES: dict[str, dict] = {
    "Example: higher-risk patient": EXAMPLE_PATIENT,
    "Example: lower-risk patient": dict(
        age=37,
        sex=1,
        cp=3,
        trestbps=130,
        chol=250,
        fbs=0,
        restecg=0,
        thalach=187,
        exang=0,
        oldpeak=3.5,
        slope=3,
        ca=0,
        thal=3,
    ),
}
DEFAULTS = dict(
    age=54,
    sex=1,
    cp=3,
    trestbps=130,
    chol=240,
    fbs=0,
    restecg=0,
    thalach=150,
    exang=0,
    oldpeak=1.0,
    slope=1,
    ca=0,
    thal=3,
)
# Measurements a patient can realistically change; used by the what-if panel.
MODIFIABLE = {"trestbps": 1, "chol": 1, "thalach": 1, "oldpeak": 0.1}


@st.cache_resource
def load_model() -> HeartModel:
    return HeartModel.load(get_settings().model_path)


@st.cache_resource
def prediction_logger() -> PredictionLogger:
    settings = get_settings()
    return PredictionLogger(settings.prediction_log_path, settings.log_predictions)


@st.cache_data
def load_metadata() -> dict:
    return json.loads(get_settings().metadata_path.read_text(encoding="utf-8"))


def patient_form(values: dict) -> dict | None:
    with st.form("patient"):
        st.subheader("Patient details")
        c1, c2, c3 = st.columns(3)

        c1.markdown("**Demographics & vitals**")
        data = {
            "age": number_input(c1, "age", values["age"]),
            "sex": category_input(c1, "sex", values["sex"]),
            "trestbps": number_input(c1, "trestbps", values["trestbps"]),
            "chol": number_input(c1, "chol", values["chol"]),
            "fbs": category_input(c1, "fbs", values["fbs"]),
        }
        c2.markdown("**Symptoms & ECG**")
        data |= {
            "cp": category_input(c2, "cp", values["cp"]),
            "restecg": category_input(c2, "restecg", values["restecg"]),
            "exang": category_input(c2, "exang", values["exang"]),
        }
        c3.markdown("**Stress test & imaging**")
        data |= {
            "thalach": number_input(c3, "thalach", values["thalach"]),
            "oldpeak": number_input(c3, "oldpeak", values["oldpeak"], step=0.1),
            "slope": category_input(c3, "slope", values["slope"]),
            "ca": number_input(c3, "ca", values["ca"]),
            "thal": category_input(c3, "thal", values["thal"]),
        }
        submitted = st.form_submit_button("Assess risk", type="primary")
    return data if submitted else None


@dataclass(frozen=True)
class Assessment:
    """Everything shown for one submitted form. Rendered once, then reused across reruns
    (e.g. when what-if sliders move) so charts and the PDF are not rebuilt every time."""

    patient: PatientInput
    prediction: Prediction
    run_id: str
    waterfall: bytes
    pdf: bytes


def show_result(a: Assessment) -> None:
    patient, prediction = a.patient, a.prediction
    st.divider()
    gauge_col, summary_col = st.columns([3, 2])
    with gauge_col:
        st.plotly_chart(risk_gauge(prediction.probability, prediction.threshold))
    with summary_col:
        color = BAND_COLORS[prediction.risk_band]
        st.markdown(
            f"<h3 style='margin-bottom:0'>Risk band: "
            f"<span style='color:{color}'>{prediction.risk_band}</span></h3>",
            unsafe_allow_html=True,
        )
        screen = "Positive" if prediction.has_disease else "Negative"
        st.metric(
            "Screening result",
            screen,
            help=f"Positive when probability ≥ {prediction.threshold:.0%} (tuned for 90% recall)",
        )
        st.caption(f"Average training patient: {prediction.base_probability:.0%}")

    st.subheader("Why this estimate?")
    st.caption(
        "SHAP values show how each measurement moved the estimate away from the average "
        "training patient. Units are percentage points."
    )
    left, right = st.columns([3, 2])
    left.image(a.waterfall)
    with right:
        factor_lists(prediction)
        st.download_button(
            "Download PDF report",
            a.pdf,
            file_name=f"heart_risk_report_{datetime.now():%Y%m%d_%H%M}.pdf",
            mime="application/pdf",
            type="primary",
        )

    what_if(patient, prediction, a.run_id)


def what_if(patient: PatientInput, prediction: Prediction, run_id: str) -> None:
    model = load_model()
    st.subheader("What if?")
    st.caption(
        "Change the measurements a patient can influence and see how the estimate responds. "
        "This shows how the model reacts, not the effect of a treatment."
    )
    base = patient.model_dump()
    changed = dict(base)
    cols = st.columns(len(MODIFIABLE))
    for col, (feature, step) in zip(cols, MODIFIABLE.items(), strict=True):
        low, high = NUMERIC_RANGES[feature]
        cast = float if step < 1 else int
        changed[feature] = col.slider(
            FEATURE_LABELS[feature],
            min_value=cast(low),
            max_value=cast(high),
            value=cast(base[feature]),
            step=cast(step),
            key=f"whatif_{feature}_{run_id}",
        )
    scenario = model.predict(PatientInput(**changed), explain=False)
    delta = scenario.probability - prediction.probability
    c1, c2 = st.columns(2)
    c1.metric(
        "Scenario probability",
        f"{scenario.probability:.0%}",
        delta=f"{delta:+.1%}",
        delta_color="inverse",
    )
    c2.metric("Scenario risk band", scenario.risk_band)


def assessment_tab(model: HeartModel) -> None:
    choice = st.selectbox("Start from", ["Blank form (typical values)", *EXAMPLES])
    if st.session_state.get("example") != choice:
        st.session_state["example"] = choice
        st.session_state.pop("assessment", None)
    values = EXAMPLES.get(choice, DEFAULTS)

    data = patient_form(values)
    if data is not None:
        try:
            patient = PatientInput(**data)
        except ValidationError as exc:
            st.session_state.pop("assessment", None)
            st.error(f"Invalid input: {exc}")
            return
        with st.spinner("Computing explanation..."):
            prediction = model.predict(patient)
            png = waterfall_png(prediction)
            pdf = build_pdf_report(patient, prediction, png)
        run_id = uuid.uuid4().hex
        prediction_logger().log(
            request_id=run_id,
            source="ui",
            model_version=model.version,
            inputs=patient.model_dump(),
            probability=prediction.probability,
            has_disease=prediction.has_disease,
        )
        st.session_state["assessment"] = Assessment(patient, prediction, run_id, png, pdf)

    if "assessment" in st.session_state:
        show_result(st.session_state["assessment"])


def main() -> None:
    st.title("🫀 Heart Disease Risk Assessment")
    st.warning(
        "For education and research only. This is not a medical device and does not replace "
        "assessment by a qualified clinician.",
        icon="⚠️",
    )

    try:
        model = load_model()
        metadata = load_metadata()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    sidebar(model, metadata)
    assess, batch, performance = st.tabs(
        ["Risk assessment", "Batch screening", "Model performance"]
    )
    with performance:
        performance_tab(metadata, REPORTS_DIR, MODEL_CARD_PATH)
    with batch:
        batch_tab(model, prediction_logger(), get_settings().max_batch_size)
    with assess:
        assessment_tab(model)


main()
