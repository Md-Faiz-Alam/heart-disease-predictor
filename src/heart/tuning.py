"""Model families, Optuna search spaces and estimator construction."""

import logging
from collections.abc import Sequence
from typing import Any, Literal

import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from xgboost import XGBClassifier

from heart.config import CV_FOLDS, FEATURES, RANDOM_STATE
from heart.features import build_pipeline

log = logging.getLogger(__name__)

FAMILIES = ("logistic_regression", "random_forest", "xgboost", "lightgbm", "svm")
Calibration = Literal["none", "sigmoid"]


def suggest_params(trial: optuna.Trial, family: str) -> dict[str, Any]:
    match family:
        case "logistic_regression":
            return {
                "C": trial.suggest_float("C", 1e-3, 10.0, log=True),
                "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
            }
        case "random_forest":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=100),
                "max_depth": trial.suggest_categorical("max_depth", [None, 3, 4, 5, 6, 8, 10]),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
                "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
            }
        case "xgboost":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
                "max_depth": trial.suggest_int("max_depth", 2, 6),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            }
        case "lightgbm":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 4, 32),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 40),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            }
        case "svm":
            return {
                "C": trial.suggest_float("C", 1e-2, 100.0, log=True),
                "gamma": trial.suggest_float("gamma", 1e-4, 1.0, log=True),
            }
    raise ValueError(f"Unknown model family: {family}")


def make_model(family: str, params: dict[str, Any], probabilities: bool = True) -> BaseEstimator:
    """Build an unfitted classifier. n_jobs=1 because parallelism happens across CV folds."""
    match family:
        case "logistic_regression":
            return LogisticRegression(max_iter=5000, **params)
        case "random_forest":
            return RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1, **params)
        case "xgboost":
            return XGBClassifier(
                eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=1, **params
            )
        case "lightgbm":
            return LGBMClassifier(
                random_state=RANDOM_STATE, n_jobs=1, subsample_freq=1, verbose=-1, **params
            )
        case "svm":
            svc = SVC(kernel="rbf", **params)
            # SVC has no native probabilities; ROC-AUC tuning can use decision_function.
            return CalibratedClassifierCV(svc, ensemble=False) if probabilities else svc
    raise ValueError(f"Unknown model family: {family}")


def make_estimator(
    family: str,
    params: dict[str, Any],
    features: Sequence[str] = FEATURES,
    calibration: Calibration = "none",
) -> BaseEstimator:
    pipeline = build_pipeline(make_model(family, params), features)
    if calibration == "sigmoid":
        return CalibratedClassifierCV(pipeline, method="sigmoid", cv=CV_FOLDS)
    return pipeline


def tune(
    family: str,
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int,
    features: Sequence[str] = FEATURES,
) -> tuple[dict[str, Any], float]:
    """Maximise mean stratified-CV ROC-AUC with TPE. Returns (best_params, best_score)."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, family)
        pipeline = build_pipeline(make_model(family, params, probabilities=False), features)
        return cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc", n_jobs=-1).mean()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        study_name=family,
    )
    study.optimize(objective, n_trials=n_trials)
    log.info("Tuned %-20s best CV ROC-AUC %.3f  %s", family, study.best_value, study.best_params)
    return study.best_params, float(study.best_value)
