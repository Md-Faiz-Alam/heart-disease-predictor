"""Validated request/response models. Inputs the model was not trained on are rejected."""

from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from heart.config import CATEGORY_LABELS, FEATURE_LABELS, FEATURES, NUMERIC_RANGES

EXAMPLE_PATIENT: dict[str, Any] = {
    "age": 67,
    "sex": 1,
    "cp": 4,
    "trestbps": 160,
    "chol": 286,
    "fbs": 0,
    "restecg": 2,
    "thalach": 108,
    "exang": 1,
    "oldpeak": 1.5,
    "slope": 2,
    "ca": 3,
    "thal": 3,
}


def _numeric(name: str) -> Any:
    low, high = NUMERIC_RANGES[name]
    return Field(ge=low, le=high, description=f"{FEATURE_LABELS[name]} [{low}-{high}]")


def _category(name: str) -> Any:
    codes = ", ".join(f"{k} = {v}" for k, v in CATEGORY_LABELS[name].items())
    return Field(description=f"{FEATURE_LABELS[name]} ({codes})")


class PatientInput(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [EXAMPLE_PATIENT]})

    age: int = _numeric("age")
    sex: int = _category("sex")
    cp: int = _category("cp")
    trestbps: int = _numeric("trestbps")
    chol: int = _numeric("chol")
    fbs: int = _category("fbs")
    restecg: int = _category("restecg")
    thalach: int = _numeric("thalach")
    exang: int = _category("exang")
    oldpeak: float = _numeric("oldpeak")
    slope: int = _category("slope")
    ca: int = _numeric("ca")
    thal: int = _category("thal")

    @model_validator(mode="after")
    def _check_category_codes(self) -> "PatientInput":
        errors = []
        for field, labels in CATEGORY_LABELS.items():
            value = getattr(self, field)
            if value not in labels:
                errors.append(f"{field}={value} is not one of {sorted(labels)}")
        if errors:
            raise ValueError("; ".join(errors))
        return self

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([self.model_dump()], columns=FEATURES).astype(float)


def parse_patients(df: pd.DataFrame) -> tuple[list[tuple[int, PatientInput]], list[dict]]:
    """Validate each row of an uploaded table. Returns (valid rows, row-level errors).

    Extra columns (e.g. a patient_id) are ignored here and kept by the caller.
    """
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")

    valid: list[tuple[int, PatientInput]] = []
    errors: list[dict] = []
    for idx, row in df[FEATURES].iterrows():
        record = {k: (None if pd.isna(v) else v) for k, v in row.items()}
        try:
            valid.append((idx, PatientInput(**record)))
        except ValidationError as exc:
            messages = "; ".join(
                f"{'.'.join(map(str, e['loc'])) or 'row'}: {e['msg']}" for e in exc.errors()
            )
            errors.append({"row": int(idx) + 1, "error": messages})
    return valid, errors


# ---------------------------------------------------------------- API responses


class Contribution(BaseModel):
    feature: str
    label: str
    value: float
    display_value: str
    shap_value: float = Field(description="Change in predicted probability from this feature")


class PredictionResponse(BaseModel):
    request_id: str
    probability: float = Field(ge=0, le=1)
    has_disease: bool = Field(description="probability >= threshold")
    risk_band: str
    threshold: float
    base_probability: float | None = None
    top_factors: list[Contribution] = []
    model_name: str
    model_version: str


class BatchRequest(BaseModel):
    patients: list[PatientInput] = Field(min_length=1)


class BatchItem(BaseModel):
    probability: float
    has_disease: bool
    risk_band: str


class BatchResponse(BaseModel):
    request_id: str
    model_version: str
    threshold: float
    n: int
    n_positive: int
    results: list[BatchItem]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None = None


class ModelInfo(BaseModel):
    model_name: str
    model_version: str
    threshold: float
    feature_set: str
    features: list[str]
    calibration: str
    metrics: dict[str, Any] = {}
    input_schema: dict[str, Any] = {}
