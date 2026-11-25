import json

import joblib
import pytest

from heart.features import build_pipeline
from heart.predict import HeartModel
from heart.schemas import PatientInput
from heart.train import train
from heart.tuning import make_model


def test_pipeline_drops_unused_features(raw_df):
    from heart.data import split_features_target

    X, y = split_features_target(raw_df)
    subset = ["age", "cp", "thal"]
    pipe = build_pipeline(make_model("logistic_regression", {}), subset).fit(X, y)
    used = set(pipe.named_steps["preprocess"].feature_names_in_)
    assert used == set(X.columns)  # accepts all inputs...
    out_cols = pipe.named_steps["preprocess"].get_feature_names_out()
    assert not any(c.startswith(("chol", "trestbps")) for c in out_cols)  # ...uses only subset


@pytest.mark.slow
def test_train_end_to_end(tmp_path, patient_values):
    metadata = train(
        model_path=tmp_path / "model.joblib",
        metadata_path=tmp_path / "metadata.json",
        model_card_path=tmp_path / "MODEL_CARD.md",
        reports_dir=tmp_path / "reports",
        n_trials=2,
        families=["logistic_regression"],
        track=False,
    )
    assert metadata["cv"]["results"]["logistic_regression"]["roc_auc"]["mean"] > 0.8
    assert 0.0 < metadata["threshold"] < 1.0
    assert json.loads((tmp_path / "metadata.json").read_text())["version"] == metadata["version"]
    assert "Limitations" in (tmp_path / "MODEL_CARD.md").read_text(encoding="utf-8")
    assert len(list((tmp_path / "reports").glob("*.png"))) == 7

    model = HeartModel(joblib.load(tmp_path / "model.joblib"))
    assert model.threshold == metadata["threshold"]
    result = model.predict(PatientInput(**patient_values))
    total = result.base_probability + sum(c.shap_value for c in result.contributions)
    assert total == pytest.approx(result.probability, abs=1e-6)
