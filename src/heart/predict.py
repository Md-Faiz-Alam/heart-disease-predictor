"""Inference: load the saved bundle, predict and explain with SHAP."""

import threading
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap

from heart.config import CATEGORY_LABELS, FEATURE_LABELS, MODEL_PATH, RANDOM_STATE
from heart.schemas import PatientInput


@dataclass(frozen=True)
class FeatureContribution:
    feature: str
    label: str
    value: float
    display_value: str
    shap_value: float  # change in predicted probability caused by this feature


@dataclass(frozen=True)
class Prediction:
    probability: float
    has_disease: bool
    risk_band: str
    threshold: float
    base_probability: float
    contributions: list[FeatureContribution]
    model_name: str
    model_version: str


HIGH_RISK_CUTOFF = 0.6


def risk_band(probability: float, threshold: float) -> str:
    """Bands are tied to the screening threshold, so "Low" always means screen-negative."""
    if probability < threshold:
        return "Low"
    if probability < max(HIGH_RISK_CUTOFF, threshold):
        return "Moderate"
    return "High"


def display_value(feature: str, value: float) -> str:
    if feature in CATEGORY_LABELS:
        return CATEGORY_LABELS[feature].get(int(value), str(value))
    return f"{value:g}"


class HeartModel:
    def __init__(self, bundle: dict[str, Any]):
        self.pipeline = bundle["pipeline"]
        self.features: list[str] = bundle["features"]
        self.background: pd.DataFrame = bundle["shap_background"]
        self.threshold: float = bundle["threshold"]
        self.model_name: str = bundle["model_name"]
        self.version: str = bundle["version"]
        self.feature_set: str = bundle.get("feature_set", "all")
        self.model_features: list[str] = bundle.get("model_features", self.features)
        self.calibration: str = bundle.get("calibration", "none")
        self._explain_lock = threading.Lock()

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "HeartModel":
        if not path.exists():
            raise FileNotFoundError(
                f"No trained model at {path}. Train one first with: python -m heart.train"
            )
        return cls(joblib.load(path))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(X[self.features])[:, 1]

    def predict_batch(self, X: pd.DataFrame) -> pd.DataFrame:
        """Vectorised scoring without explanations (for batch screening)."""
        proba = self.predict_proba(X)
        return pd.DataFrame(
            {
                "probability": proba,
                "has_disease": proba >= self.threshold,
                "risk_band": [risk_band(p, self.threshold) for p in proba],
            },
            index=X.index,
        )

    def _proba_from_array(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(pd.DataFrame(X, columns=self.features))

    @cached_property
    def explainer(self) -> shap.Explainer:
        # Model-agnostic SHAP on the ORIGINAL clinical features (not the one-hot columns),
        # measured in probability space, against real training patients as the baseline.
        return shap.Explainer(
            self._proba_from_array,
            shap.maskers.Independent(self.background.to_numpy(), max_samples=len(self.background)),
            feature_names=self.features,
            algorithm="permutation",
            seed=RANDOM_STATE,
        )

    def explain(self, X: pd.DataFrame) -> shap.Explanation:
        # One explainer is shared by API worker threads and Streamlit sessions; its seeded
        # RNG and masker are stateful, so serialise calls (each takes well under a second).
        with self._explain_lock:
            return self.explainer(X[self.features].to_numpy())

    def predict(self, patient: PatientInput, explain: bool = True) -> Prediction:
        X = patient.to_frame()
        probability = float(self.predict_proba(X)[0])

        contributions: list[FeatureContribution] = []
        base = float(self.predict_proba(self.background).mean())
        if explain:
            explanation = self.explain(X)
            base = float(np.asarray(explanation.base_values).ravel()[0])
            for feature, value, sv in zip(
                self.features, X.iloc[0].to_numpy(), explanation.values[0], strict=True
            ):
                contributions.append(
                    FeatureContribution(
                        feature=feature,
                        label=FEATURE_LABELS[feature],
                        value=float(value),
                        display_value=display_value(feature, float(value)),
                        shap_value=float(sv),
                    )
                )
            contributions.sort(key=lambda c: abs(c.shap_value), reverse=True)

        return Prediction(
            probability=probability,
            has_disease=probability >= self.threshold,
            risk_band=risk_band(probability, self.threshold),
            threshold=self.threshold,
            base_probability=base,
            contributions=contributions,
            model_name=self.model_name,
            model_version=self.version,
        )
