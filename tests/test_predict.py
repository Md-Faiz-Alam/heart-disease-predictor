import pytest

from heart.predict import risk_band
from heart.schemas import PatientInput


def test_prediction_is_probability(trained_model, patient_values):
    result = trained_model.predict(PatientInput(**patient_values))
    assert 0.0 <= result.probability <= 1.0
    assert result.has_disease == (result.probability >= result.threshold)
    assert result.risk_band in {"Low", "Moderate", "High"}


def test_shap_values_add_up_to_prediction(trained_model, patient_values):
    result = trained_model.predict(PatientInput(**patient_values))
    total = result.base_probability + sum(c.shap_value for c in result.contributions)
    assert total == pytest.approx(result.probability, abs=1e-6)


def test_shap_explains_original_features(trained_model, patient_values):
    result = trained_model.predict(PatientInput(**patient_values))
    assert {c.feature for c in result.contributions} == set(patient_values)
    cp = next(c for c in result.contributions if c.feature == "cp")
    assert cp.display_value == "Asymptomatic"


def test_higher_risk_profile_scores_higher(trained_model, patient_values):
    low = dict(patient_values, cp=3, thalach=187, exang=0, ca=0, oldpeak=0.0, slope=1, age=37)
    high_p = trained_model.predict(PatientInput(**patient_values), explain=False).probability
    low_p = trained_model.predict(PatientInput(**low), explain=False).probability
    assert high_p > low_p


@pytest.mark.parametrize(
    ("p", "threshold", "band"),
    [
        (0.1, 0.3, "Low"),
        (0.29, 0.3, "Low"),
        (0.3, 0.3, "Moderate"),
        (0.9, 0.3, "High"),
        (0.65, 0.7, "Low"),
        (0.75, 0.7, "High"),
    ],
)
def test_risk_band_tied_to_threshold(p, threshold, band):
    assert risk_band(p, threshold) == band


def test_low_band_always_means_screen_negative(trained_model, raw_df):
    from heart.data import split_features_target

    X, _ = split_features_target(raw_df)
    batch = trained_model.predict_batch(X.dropna())
    low = batch[batch["risk_band"] == "Low"]
    assert not low["has_disease"].any()
    assert batch.loc[~batch["has_disease"], "risk_band"].eq("Low").all()
