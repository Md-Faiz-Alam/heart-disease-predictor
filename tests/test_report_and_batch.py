import json
import logging

import pandas as pd
import pytest

from heart.observability import JsonFormatter, PredictionLogger
from heart.report import build_pdf_report
from heart.schemas import PatientInput, parse_patients
from heart.ui.batch import template_csv


def test_pdf_report_is_valid_pdf(trained_model, patient_values):
    patient = PatientInput(**patient_values)
    pdf = build_pdf_report(patient, trained_model.predict(patient))
    assert pdf.startswith(b"%PDF")
    assert len(pdf) > 1000


def test_template_csv_parses_cleanly():
    from io import BytesIO

    df = pd.read_csv(BytesIO(template_csv()))
    valid, errors = parse_patients(df)
    assert len(valid) == 2
    assert errors == []


def test_parse_patients_reports_bad_rows(patient_values):
    df = pd.DataFrame([patient_values, dict(patient_values, cp=0), dict(patient_values, age=None)])
    valid, errors = parse_patients(df)
    assert [i for i, _ in valid] == [0]
    assert [e["row"] for e in errors] == [2, 3]
    assert "cp" in errors[0]["error"]


def test_parse_patients_requires_columns(patient_values):
    df = pd.DataFrame([patient_values]).drop(columns=["thal"])
    with pytest.raises(ValueError, match="thal"):
        parse_patients(df)


def test_prediction_logger_disabled_writes_nothing(tmp_path):
    path = tmp_path / "log.jsonl"
    PredictionLogger(path, enabled=False).log(
        request_id="x",
        source="t",
        model_version="v",
        inputs={},
        probability=0.5,
        has_disease=True,
    )
    assert not path.exists()


def test_json_formatter_includes_extra_fields():
    record = logging.LogRecord("t", logging.INFO, "", 0, "hello", None, None)
    record.request_id = "abc"
    payload = json.loads(JsonFormatter().format(record))
    assert payload["msg"] == "hello"
    assert payload["request_id"] == "abc"
