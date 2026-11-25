"""Quality gates on the committed model artifact. CI fails if a worse or stale model is shipped."""

import json

import joblib
import pytest
import sklearn

from heart.config import FEATURES, METADATA_PATH, MIN_CV_ROC_AUC, MODEL_PATH, TARGET_RECALL
from heart.train import QualityGateError, train

pytestmark = pytest.mark.skipif(not MODEL_PATH.exists(), reason="no trained model")


@pytest.fixture(scope="module")
def metadata() -> dict:
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


def test_cv_roc_auc_above_gate(metadata):
    cv = metadata["cv"]["results"][metadata["model_name"]]
    assert cv["roc_auc"]["mean"] >= MIN_CV_ROC_AUC


def test_threshold_reaches_target_recall_out_of_fold(metadata):
    # Subgroups are computed out-of-fold on the training set at the chosen threshold.
    groups = metadata["subgroups_oof"]
    n = sum(groups[g]["n"] for g in ("sex: female", "sex: male"))
    pos = {g: groups[g]["n"] * groups[g]["positive_rate"] for g in ("sex: female", "sex: male")}
    recall = sum(pos[g] * groups[g]["recall"] for g in pos) / sum(pos.values())
    assert n == metadata["data"]["n_train"]
    assert recall >= TARGET_RECALL - 1e-9


def test_artifact_matches_metadata_and_environment(metadata):
    bundle = joblib.load(MODEL_PATH)
    assert bundle["version"] == metadata["version"]
    assert bundle["features"] == FEATURES
    assert bundle["threshold"] == metadata["threshold"]
    # A model pickled with a different sklearn version may silently change behaviour.
    assert bundle["sklearn_version"] == sklearn.__version__, (
        f"Model trained with sklearn {bundle['sklearn_version']}, running {sklearn.__version__}. "
        "Retrain with: python -m heart.train"
    )


@pytest.mark.slow
def test_training_refuses_to_save_below_gate(tmp_path):
    with pytest.raises(QualityGateError):
        train(
            model_path=tmp_path / "m.joblib",
            metadata_path=tmp_path / "m.json",
            model_card_path=tmp_path / "card.md",
            reports_dir=tmp_path / "reports",
            n_trials=1,
            families=["logistic_regression"],
            track=False,
            min_cv_roc_auc=0.999,
        )
    assert not (tmp_path / "m.joblib").exists()
