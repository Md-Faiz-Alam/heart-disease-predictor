"""End-to-end training: tune -> compare -> select features -> calibrate -> threshold -> report.

Usage:
    python -m heart.train                     # full run (Optuna, all model families)
    python -m heart.train --trials 10         # quicker
    python -m heart.train --no-track          # skip MLflow
"""

import argparse
import json
import logging
import platform
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)

from heart import evaluate
from heart.config import (
    CV_FOLDS,
    CV_REPEATS,
    DEFAULT_THRESHOLD,
    FEATURE_SETS,
    FEATURES,
    METADATA_PATH,
    MIN_CV_ROC_AUC,
    MODEL_CARD_PATH,
    MODEL_PATH,
    N_TRIALS,
    RANDOM_STATE,
    REPORTS_DIR,
    SHAP_BACKGROUND_SIZE,
    TARGET_RECALL,
    TEST_SIZE,
)
from heart.data import load_raw, split_features_target
from heart.predict import HeartModel
from heart.tracking import mlflow_run
from heart.tuning import FAMILIES, Calibration, make_estimator, tune

log = logging.getLogger(__name__)


class QualityGateError(RuntimeError):
    """The trained model is not good enough to replace the current one."""


CV_SCORING = ["roc_auc", "recall", "precision", "f1", "accuracy", "neg_brier_score"]


def repeated_cv(
    estimator: BaseEstimator, X: pd.DataFrame, y: pd.Series, repeats: int = CV_REPEATS
) -> dict[str, dict[str, float]]:
    """Mean/std of each metric over repeated stratified K-fold.

    Uses a different seed from tuning so the comparison is not on the exact tuning splits.
    """
    cv = RepeatedStratifiedKFold(
        n_splits=CV_FOLDS, n_repeats=repeats, random_state=RANDOM_STATE + 1
    )
    scores = cross_validate(estimator, X, y, cv=cv, scoring=CV_SCORING, n_jobs=-1)
    out: dict[str, dict[str, float]] = {}
    for metric in CV_SCORING:
        values = scores[f"test_{metric}"]
        sign = -1 if metric.startswith("neg_") else 1
        out[metric.removeprefix("neg_")] = {
            "mean": float(sign * values.mean()),
            "std": float(values.std()),
        }
    return out


def _best(results: dict[str, dict], metric: str = "roc_auc", lower_is_better: bool = False) -> str:
    sign = -1 if lower_is_better else 1
    return max(results, key=lambda k: sign * results[k][metric]["mean"])


def make_bundle(
    estimator: BaseEstimator,
    X_train: pd.DataFrame,
    model_name: str,
    version: str,
    threshold: float = DEFAULT_THRESHOLD,
    feature_set: str = "all",
    calibration: str = "none",
) -> dict[str, Any]:
    """Everything needed at inference time, saved as one artifact."""
    complete = X_train[FEATURES].dropna()
    background = complete.sample(
        n=min(SHAP_BACKGROUND_SIZE, len(complete)), random_state=RANDOM_STATE
    )
    return {
        "pipeline": estimator,
        "features": FEATURES,  # inputs the model accepts
        "model_features": FEATURE_SETS[feature_set],  # inputs the model actually uses
        "feature_set": feature_set,
        "calibration": calibration,
        "shap_background": background.reset_index(drop=True),
        "threshold": threshold,
        "model_name": model_name,
        "version": version,
        "sklearn_version": sklearn.__version__,
    }


def _generate_reports(
    reports_dir: Path,
    model: HeartModel,
    cv_results: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    test_proba: np.ndarray,
    y_train: pd.Series,
    oof_proba: np.ndarray,
    threshold: float,
) -> list[Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    for old in reports_dir.glob("*.png"):
        old.unlink()
    return [
        evaluate.plot_model_comparison(cv_results, reports_dir / "model_comparison.png"),
        evaluate.plot_roc(y_test, test_proba, reports_dir / "roc_curve.png"),
        evaluate.plot_precision_recall(y_test, test_proba, threshold, reports_dir / "pr_curve.png"),
        evaluate.plot_calibration(y_train, oof_proba, reports_dir / "calibration.png"),
        evaluate.plot_threshold_tradeoff(
            y_train, oof_proba, threshold, TARGET_RECALL, reports_dir / "threshold_tradeoff.png"
        ),
        evaluate.plot_confusion(
            y_test, test_proba, threshold, reports_dir / "confusion_matrix.png"
        ),
        evaluate.plot_shap_summary(model.explain(X_test), reports_dir / "shap_summary.png"),
    ]


def _log_to_mlflow(mlflow, metadata: dict, artifacts: list[Path]) -> None:
    best = metadata["model_name"]
    mlflow.log_params(
        {
            "model_family": best,
            "feature_set": metadata["feature_set"],
            "calibration": metadata["calibration"],
            "threshold": metadata["threshold"],
            "target_recall": metadata["target_recall"],
            "n_trials": metadata["tuning"]["n_trials"],
            **{f"hp.{k}": v for k, v in metadata["tuning"]["results"][best]["params"].items()},
        }
    )
    mlflow.log_metrics(
        {
            "cv_roc_auc": metadata["cv"]["results"][best]["roc_auc"]["mean"],
            "cv_brier": metadata["cv"]["results"][best]["brier_score"]["mean"],
            **{f"test_{k}": v for k, v in metadata["test_metrics"].items() if k != "n_samples"},
        }
    )
    for family, result in metadata["cv"]["results"].items():
        with mlflow.start_run(run_name=family, nested=True):
            mlflow.log_params(metadata["tuning"]["results"][family]["params"])
            mlflow.log_metrics(
                {
                    "cv_roc_auc": result["roc_auc"]["mean"],
                    "cv_roc_auc_std": result["roc_auc"]["std"],
                    "cv_recall": result["recall"]["mean"],
                    "cv_brier": result["brier_score"]["mean"],
                }
            )
    for path in artifacts:
        mlflow.log_artifact(str(path))


def train(
    model_path: Path = MODEL_PATH,
    metadata_path: Path = METADATA_PATH,
    model_card_path: Path = MODEL_CARD_PATH,
    reports_dir: Path = REPORTS_DIR,
    n_trials: int = N_TRIALS,
    families: Sequence[str] = FAMILIES,
    track: bool = True,
    min_cv_roc_auc: float = MIN_CV_ROC_AUC,
) -> dict:
    df = load_raw()
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    log.info(
        "Train %d rows, held-out test %d rows (test set untouched until the end)",
        len(X_train),
        len(X_test),
    )

    # 1. Tune every family, then compare them all on the same repeated CV.
    tuning: dict[str, dict[str, Any]] = {}
    for family in families:
        params, score = tune(family, X_train, y_train, n_trials)
        tuning[family] = {"params": params, "tuning_cv_roc_auc": score}
    cv_results = {
        family: repeated_cv(make_estimator(family, tuning[family]["params"]), X_train, y_train)
        for family in families
    }
    for family, r in cv_results.items():
        log.info(
            "CV %-20s ROC-AUC %.3f ± %.3f  Brier %.3f",
            family,
            r["roc_auc"]["mean"],
            r["roc_auc"]["std"],
            r["brier_score"]["mean"],
        )
    best = _best(cv_results)
    params = tuning[best]["params"]
    log.info("Selected family: %s", best)
    best_auc = cv_results[best]["roc_auc"]["mean"]
    if best_auc < min_cv_roc_auc:
        raise QualityGateError(
            f"Best CV ROC-AUC {best_auc:.3f} is below the gate of {min_cv_roc_auc:.3f}; "
            "refusing to save a model."
        )

    # 2. Feature set: the EDA subset must beat all features on CV to be used.
    feature_set_results = {
        name: repeated_cv(make_estimator(best, params, feats), X_train, y_train)
        for name, feats in FEATURE_SETS.items()
    }
    feature_set = _best(feature_set_results)
    features = FEATURE_SETS[feature_set]
    log.info(
        "Feature sets: %s -> %s",
        {k: round(v["roc_auc"]["mean"], 3) for k, v in feature_set_results.items()},
        feature_set,
    )

    # 3. Calibration: keep sigmoid calibration only if it lowers the Brier score.
    # SVM is already calibrated internally.
    options: list[Calibration] = ["none"] if best == "svm" else ["none", "sigmoid"]
    calibration_results = {
        c: repeated_cv(make_estimator(best, params, features, c), X_train, y_train, repeats=1)
        for c in options
    }
    calibration: Calibration = min(
        options, key=lambda c: calibration_results[c]["brier_score"]["mean"]
    )
    log.info(
        "Calibration: %s -> %s",
        {k: round(v["brier_score"]["mean"], 4) for k, v in calibration_results.items()},
        calibration,
    )

    # 4. Threshold from out-of-fold probabilities on the training set.
    estimator = make_estimator(best, params, features, calibration)
    oof_proba = cross_val_predict(
        clone(estimator),
        X_train,
        y_train,
        cv=StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]
    threshold = evaluate.choose_threshold(y_train, oof_proba, TARGET_RECALL)
    log.info("Decision threshold %.3f (target recall %.0f%%)", threshold, TARGET_RECALL * 100)

    # 5. Final fit and a single look at the held-out test set.
    estimator.fit(X_train, y_train)
    test_proba = estimator.predict_proba(X_test)[:, 1]
    test_metrics = evaluate.classification_metrics(y_test, test_proba, threshold)
    log.info("Test: %s", {k: round(v, 3) for k, v in test_metrics.items()})

    version = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    bundle = make_bundle(estimator, X_train, best, version, threshold, feature_set, calibration)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)

    metadata = {
        "version": version,
        "model_name": best,
        "selection_metric": "repeated CV ROC-AUC",
        "feature_set": feature_set,
        "feature_sets": FEATURE_SETS,
        "calibration": calibration,
        "threshold": threshold,
        "target_recall": TARGET_RECALL,
        "features": FEATURES,
        "data": {
            "n_rows": int(len(df)),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "positive_rate": float(np.round(y.mean(), 4)),
        },
        "tuning": {"sampler": "TPE", "n_trials": n_trials, "results": tuning},
        "cv": {
            "scheme": f"RepeatedStratifiedKFold({CV_FOLDS} folds x {CV_REPEATS} repeats)",
            "results": cv_results,
        },
        "feature_set_comparison": feature_set_results,
        "calibration_comparison": calibration_results,
        "test_metrics": test_metrics,
        "test_metrics_default_threshold": evaluate.classification_metrics(
            y_test, test_proba, DEFAULT_THRESHOLD
        ),
        "test_ci_95": evaluate.bootstrap_ci(y_test, test_proba, threshold),
        "subgroups_oof": evaluate.subgroup_metrics(X_train, y_train, oof_proba, threshold),
        "environment": {"python": platform.python_version(), "sklearn": sklearn.__version__},
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    evaluate.write_model_card(metadata, model_card_path)

    plots = _generate_reports(
        reports_dir,
        HeartModel(bundle),
        cv_results,
        X_test,
        y_test,
        test_proba,
        y_train,
        oof_proba,
        threshold,
    )
    log.info("Saved model, metadata, model card and %d report plots", len(plots))

    with mlflow_run(track, run_name=f"{best}-{version}") as mlflow:
        if mlflow is not None:
            _log_to_mlflow(mlflow, metadata, [metadata_path, model_card_path, model_path, *plots])
            log.info(
                "Logged run to MLflow (view: mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db)"
            )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the heart disease model")
    parser.add_argument("--trials", type=int, default=N_TRIALS, help="Optuna trials per family")
    parser.add_argument("--families", nargs="+", default=list(FAMILIES), choices=FAMILIES)
    parser.add_argument("--no-track", action="store_true", help="disable MLflow tracking")
    parser.add_argument(
        "--min-roc-auc", type=float, default=MIN_CV_ROC_AUC, help="quality gate on CV ROC-AUC"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("alembic", "mlflow", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    try:
        train(
            n_trials=args.trials,
            families=args.families,
            track=not args.no_track,
            min_cv_roc_auc=args.min_roc_auc,
        )
    except QualityGateError as exc:
        log.error(str(exc))
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
