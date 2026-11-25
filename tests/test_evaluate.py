import numpy as np
import pandas as pd
import pytest

from heart import evaluate


@pytest.fixture
def scores():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    proba = np.clip(y * 0.4 + rng.uniform(0, 0.6, 300), 0, 1)
    return y, proba


def test_choose_threshold_meets_target_recall(scores):
    y, proba = scores
    threshold = evaluate.choose_threshold(y, proba, target_recall=0.9)
    recall = evaluate.classification_metrics(y, proba, threshold)["recall"]
    assert recall >= 0.9


def test_choose_threshold_is_the_highest_meeting_target(scores):
    y, proba = scores
    threshold = evaluate.choose_threshold(y, proba, target_recall=0.9)
    slightly_higher = evaluate.classification_metrics(y, proba, threshold + 0.01)["recall"]
    assert slightly_higher < 0.9 or threshold >= proba.max()


def test_bootstrap_ci_contains_point_estimate(scores):
    y, proba = scores
    point = evaluate.classification_metrics(y, proba, 0.5)
    ci = evaluate.bootstrap_ci(y, proba, 0.5, n_boot=300)
    for metric in ("roc_auc", "recall", "precision"):
        low, high = ci[metric]
        assert low <= point[metric] <= high


def test_subgroup_metrics_cover_all_rows(scores):
    y, proba = scores
    X = pd.DataFrame({"sex": np.arange(300) % 2, "age": np.linspace(30, 80, 300)})
    groups = evaluate.subgroup_metrics(X, y, proba, 0.5)
    assert groups["sex: female"]["n"] + groups["sex: male"]["n"] == 300
    assert groups["age < 55"]["n"] + groups["age >= 55"]["n"] == 300
