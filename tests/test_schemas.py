import pytest
from pydantic import ValidationError

from heart.config import FEATURES
from heart.schemas import PatientInput


def test_valid_patient(patient_values):
    frame = PatientInput(**patient_values).to_frame()
    assert list(frame.columns) == FEATURES
    assert frame.shape == (1, len(FEATURES))


@pytest.mark.parametrize(
    ("field", "value"),
    [("cp", 0), ("thal", 1), ("thal", 2), ("slope", 0), ("sex", 2)],
)
def test_rejects_codes_the_model_never_saw(patient_values, field, value):
    patient_values[field] = value
    with pytest.raises(ValidationError):
        PatientInput(**patient_values)


@pytest.mark.parametrize(("field", "value"), [("age", 5), ("chol", 2000), ("oldpeak", -1)])
def test_rejects_out_of_range(patient_values, field, value):
    patient_values[field] = value
    with pytest.raises(ValidationError):
        PatientInput(**patient_values)


def test_rejects_unknown_fields(patient_values):
    with pytest.raises(ValidationError):
        PatientInput(**patient_values, bmi=30)
