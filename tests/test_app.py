"""Streamlit smoke tests using the real trained model in models/."""

import pytest
from streamlit.testing.v1 import AppTest

from heart.config import MODEL_PATH, ROOT

pytestmark = pytest.mark.skipif(not MODEL_PATH.exists(), reason="no trained model")


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("HEART_PREDICTION_LOG_PATH", str(tmp_path / "predictions.jsonl"))
    from heart.settings import get_settings

    get_settings.cache_clear()
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    yield at
    get_settings.cache_clear()


def test_app_renders_without_errors(app):
    app.run()
    assert not app.exception
    assert [t.label for t in app.tabs] == [
        "Risk assessment",
        "Batch screening",
        "Model performance",
    ]


def test_assessment_flow(app):
    app.run()
    app.selectbox[0].set_value("Example: higher-risk patient").run()
    app.button[0].click().run()
    assert not app.exception
    assert any("Screening result" in m.label for m in app.metric)
    assert any("Scenario probability" in m.label for m in app.metric)
