import pytest
from fastapi.testclient import TestClient

from heart.api import create_app
from heart.serve import build_command, healthcheck
from heart.settings import Settings


def test_api_is_default_and_uses_platform_port():
    cmd = build_command({"PORT": "10000"})
    assert cmd[:2] == ["uvicorn", "heart.api:app"]
    assert cmd[cmd.index("--port") + 1] == "10000"
    assert cmd[cmd.index("--workers") + 1] == "1"


def test_ui_service_and_default_port():
    cmd = build_command({"SERVICE": "UI"})
    assert cmd[:2] == ["streamlit", "run"]
    assert cmd[2].endswith("app.py")
    assert "--server.port=8501" in cmd
    assert "--server.address=0.0.0.0" in cmd


def test_unknown_service_fails_fast():
    with pytest.raises(SystemExit, match="SERVICE"):
        build_command({"SERVICE": "worker"})


def test_healthcheck_fails_when_nothing_listens():
    assert healthcheck({"SERVICE": "api", "PORT": "1"}) == 1


# ---------------------------------------------------------------- API key


@pytest.fixture
def secured_client(trained_model, tmp_path):
    settings = Settings(api_key="s3cret", prediction_log_path=tmp_path / "p.jsonl", log_json=False)
    with TestClient(create_app(settings=settings, model=trained_model, metadata={})) as c:
        yield c


def test_predict_requires_api_key_when_configured(secured_client, patient_values):
    assert secured_client.post("/predict", json=patient_values).status_code == 401
    wrong = secured_client.post("/predict", json=patient_values, headers={"X-API-Key": "nope"})
    assert wrong.status_code == 401
    ok = secured_client.post("/predict", json=patient_values, headers={"X-API-Key": "s3cret"})
    assert ok.status_code == 200


def test_batch_requires_api_key(secured_client, patient_values):
    body = {"patients": [patient_values]}
    assert secured_client.post("/predict/batch", json=body).status_code == 401


def test_health_and_docs_stay_public(secured_client):
    assert secured_client.get("/health").status_code == 200
    assert secured_client.get("/openapi.json").status_code == 200
