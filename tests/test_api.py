import json

import pytest
from fastapi.testclient import TestClient

from heart.api import create_app
from heart.settings import Settings


@pytest.fixture
def log_path(tmp_path):
    return tmp_path / "predictions.jsonl"


@pytest.fixture
def client(trained_model, log_path):
    settings = Settings(prediction_log_path=log_path, log_json=False, max_batch_size=5)
    app = create_app(settings=settings, model=trained_model, metadata={})
    with TestClient(app) as c:
        yield c


def test_health(client):
    body = client.get("/health").json()
    assert body == {"status": "ok", "model_loaded": True, "model_version": "test"}


def test_model_info(client):
    body = client.get("/model-info").json()
    assert body["model_name"] == "logistic_regression"
    assert 0 < body["threshold"] < 1
    assert "cp" in body["input_schema"]["properties"]


def test_predict_returns_probability_and_factors(client, patient_values):
    response = client.post("/predict?top_k=3", json=patient_values)
    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["probability"] <= 1
    assert len(body["top_factors"]) == 3
    assert body["request_id"] == response.headers["X-Request-ID"]


def test_predict_without_explanation(client, patient_values):
    body = client.post("/predict?explain=false", json=patient_values).json()
    assert body["top_factors"] == []
    assert body["base_probability"] is None


def test_request_id_is_propagated(client, patient_values):
    response = client.post("/predict", json=patient_values, headers={"X-Request-ID": "abc123"})
    assert response.headers["X-Request-ID"] == "abc123"
    assert response.json()["request_id"] == "abc123"


@pytest.mark.parametrize(("field", "value"), [("cp", 0), ("age", 500), ("thal", 2)])
def test_invalid_input_is_422(client, patient_values, field, value):
    patient_values[field] = value
    assert client.post("/predict", json=patient_values).status_code == 422


def test_unknown_field_is_422(client, patient_values):
    assert client.post("/predict", json={**patient_values, "name": "x"}).status_code == 422


def test_batch(client, patient_values):
    low = dict(patient_values, cp=3, thalach=187, exang=0, ca=0, oldpeak=0.0, age=37)
    body = client.post("/predict/batch", json={"patients": [patient_values, low]}).json()
    assert body["n"] == 2
    assert body["results"][0]["probability"] > body["results"][1]["probability"]


def test_batch_size_limit(client, patient_values):
    response = client.post("/predict/batch", json={"patients": [patient_values] * 6})
    assert response.status_code == 413


def test_predictions_are_audit_logged(client, patient_values, log_path):
    client.post("/predict", json=patient_values, headers={"X-Request-ID": "r1"})
    record = json.loads(log_path.read_text().strip().splitlines()[-1])
    assert record["request_id"] == "r1"
    assert record["source"] == "api"
    assert record["inputs"] == patient_values
