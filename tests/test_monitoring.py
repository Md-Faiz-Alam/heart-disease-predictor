import json

import numpy as np
import pandas as pd

from heart.config import FEATURES
from heart.data import split_features_target
from heart.monitoring import drift_report, main, psi


def test_psi_zero_for_identical_distributions():
    s = pd.Series(np.random.default_rng(0).normal(size=500))
    assert psi(s, s, categorical=False) < 1e-6


def test_psi_detects_shift():
    rng = np.random.default_rng(0)
    ref = pd.Series(rng.normal(0, 1, 1000))
    shifted = pd.Series(rng.normal(1.5, 1, 1000))
    assert psi(ref, shifted, categorical=False) > 0.25


def test_categorical_psi_handles_unseen_category():
    ref = pd.Series([1, 1, 2, 2])
    cur = pd.Series([1, 2, 3, 3])
    assert psi(ref, cur, categorical=True) > 0.25


def test_training_data_has_no_drift_against_itself(raw_df):
    X, _ = split_features_target(raw_df)
    assert all(r.status == "ok" for r in drift_report(X, X))


def test_cli_flags_drift_and_writes_report(raw_df, tmp_path):
    X, _ = split_features_target(raw_df)
    shifted = X.dropna().head(60).copy()
    shifted["age"] = shifted["age"] + 25  # an older population
    log = tmp_path / "predictions.jsonl"
    with log.open("w") as f:
        for row in shifted.to_dict("records"):
            f.write(json.dumps({"inputs": row}) + "\n")

    out = tmp_path / "drift.json"
    assert main(["--log", str(log), "--output", str(out), "--fail-on-drift"]) == 1
    report = json.loads(out.read_text())
    assert report["features"][0]["feature"] == "age"
    assert {f["feature"] for f in report["features"]} == set(FEATURES)
