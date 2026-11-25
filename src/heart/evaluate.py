"""Metrics, threshold selection, uncertainty, subgroup checks, report plots and model card."""

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless: training runs in terminals / CI

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from heart.config import FEATURE_LABELS, RANDOM_STATE

# ---------------------------------------------------------------- metrics


def classification_metrics(y, proba, threshold: float) -> dict[str, float]:
    y = np.asarray(y)
    proba = np.asarray(proba)
    pred = (proba >= threshold).astype(int)
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    return {
        "roc_auc": float(roc_auc_score(y, proba)),
        "average_precision": float(average_precision_score(y, proba)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "f1": float(f1_score(y, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
        "brier_score": float(brier_score_loss(y, proba)),
        "n_samples": int(len(y)),
    }


def bootstrap_ci(
    y, proba, threshold: float, n_boot: int = 1000, seed: int = RANDOM_STATE
) -> dict[str, list[float]]:
    """95% percentile bootstrap intervals. With ~60 test patients these are wide, by design."""
    y = np.asarray(y)
    proba = np.asarray(proba)
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {
        "roc_auc": [],
        "recall": [],
        "precision": [],
        "specificity": [],
    }
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        m = classification_metrics(y[idx], proba[idx], threshold)
        for key in samples:
            samples[key].append(m[key])
    return {
        k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in samples.items()
    }


def choose_threshold(y, proba, target_recall: float) -> float:
    """Highest threshold whose recall still reaches the target (= best precision at that recall)."""
    _, recall, thresholds = precision_recall_curve(y, proba)
    meets_target = recall[:-1] >= target_recall
    chosen = thresholds[meets_target].max() if meets_target.any() else thresholds.min()
    # Round DOWN: rounding up could push the threshold past the cut-off and lose recall.
    return float(np.floor(chosen * 1000) / 1000)


def subgroup_metrics(X: pd.DataFrame, y, proba, threshold: float) -> dict[str, dict[str, Any]]:
    y = np.asarray(y)
    proba = np.asarray(proba)
    groups = {
        "sex: female": (X["sex"] == 0).to_numpy(),
        "sex: male": (X["sex"] == 1).to_numpy(),
        "age < 55": (X["age"] < 55).to_numpy(),
        "age >= 55": (X["age"] >= 55).to_numpy(),
    }
    out: dict[str, dict[str, Any]] = {}
    for name, mask in groups.items():
        yy, pp = y[mask], proba[mask]
        pred = (pp >= threshold).astype(int)
        out[name] = {
            "n": int(mask.sum()),
            "positive_rate": float(yy.mean()),
            "roc_auc": float(roc_auc_score(yy, pp)) if len(np.unique(yy)) == 2 else None,
            "recall": float(recall_score(yy, pred, zero_division=0)),
            "precision": float(precision_score(yy, pred, zero_division=0)),
        }
    return out


# ---------------------------------------------------------------- plots


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_model_comparison(cv_results: dict, path: Path) -> Path:
    names = sorted(cv_results, key=lambda n: cv_results[n]["roc_auc"]["mean"])
    means = [cv_results[n]["roc_auc"]["mean"] for n in names]
    stds = [cv_results[n]["roc_auc"]["std"] for n in names]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh([n.replace("_", " ") for n in names], means, xerr=stds, color="#4C72B0", capsize=4)
    for i, (m, s) in enumerate(zip(means, stds, strict=True)):
        ax.text(m + s + 0.005, i, f"{m:.3f}", va="center", fontsize=9)
    ax.set_xlim(max(0.5, min(means) - 0.1), 1.0)
    ax.set_xlabel("ROC-AUC (mean ± std, repeated stratified CV)")
    ax.set_title("Tuned models: cross-validated ROC-AUC")
    return _save(fig, path)


def plot_roc(y, proba, path: Path) -> Path:
    fpr, tpr, _ = roc_curve(y, proba)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc_score(y, proba):.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve (test set)")
    ax.legend(loc="lower right")
    return _save(fig, path)


def plot_precision_recall(y, proba, threshold: float, path: Path) -> Path:
    precision, recall, _ = precision_recall_curve(y, proba)
    pred = np.asarray(proba) >= threshold
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(recall, precision, lw=2, label=f"AP = {average_precision_score(y, proba):.3f}")
    ax.scatter(
        [recall_score(y, pred)],
        [precision_score(y, pred, zero_division=0)],
        color="crimson",
        zorder=3,
        label=f"threshold = {threshold:.2f}",
    )
    ax.axhline(np.mean(y), color="grey", ls="--", lw=1, label="prevalence")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-recall (test set)", ylim=(0, 1.02))
    ax.legend(loc="lower left")
    return _save(fig, path)


def plot_calibration(y, proba, path: Path, label: str = "out-of-fold (train)") -> Path:
    frac_pos, mean_pred = calibration_curve(y, proba, n_bins=8, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfectly calibrated")
    ax.plot(mean_pred, frac_pos, "o-", lw=2, label=label)
    ax.set(
        xlabel="Predicted probability",
        ylabel="Observed frequency",
        title=f"Calibration (Brier = {brier_score_loss(y, proba):.3f})",
    )
    ax.legend(loc="upper left")
    return _save(fig, path)


def plot_threshold_tradeoff(y, proba, threshold: float, target_recall: float, path: Path) -> Path:
    precision, recall, thresholds = precision_recall_curve(y, proba)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(thresholds, recall[:-1], lw=2, label="recall")
    ax.plot(thresholds, precision[:-1], lw=2, label="precision")
    ax.axvline(threshold, color="crimson", ls="--", label=f"chosen = {threshold:.2f}")
    ax.axhline(target_recall, color="grey", ls=":", label=f"target recall = {target_recall:.0%}")
    ax.set(xlabel="Decision threshold", ylabel="Score", title="Threshold trade-off (out-of-fold)")
    ax.legend(loc="lower left", fontsize=8)
    return _save(fig, path)


def plot_confusion(y, proba, threshold: float, path: Path) -> Path:
    pred = (np.asarray(proba) >= threshold).astype(int)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y, pred, display_labels=["No disease", "Disease"], cmap="Blues", ax=ax, colorbar=False
    )
    ax.set_title(f"Confusion matrix (test, threshold {threshold:.2f})")
    return _save(fig, path)


def plot_shap_summary(explanation: shap.Explanation, path: Path) -> Path:
    explanation.feature_names = [FEATURE_LABELS[f] for f in explanation.feature_names]
    plt.close("all")
    shap.plots.beeswarm(explanation, max_display=13, show=False)
    fig = plt.gcf()
    fig.set_size_inches(8, 6)
    plt.title("SHAP: impact on predicted probability (test set)")
    return _save(fig, path)


# ---------------------------------------------------------------- model card


def _fmt(m: dict | None, key: str = "mean") -> str:
    return "n/a" if m is None else f"{m[key]:.3f}"


def write_model_card(metadata: dict, path: Path) -> Path:
    best = metadata["model_name"]
    cv = metadata["cv"]["results"]
    test = metadata["test_metrics"]
    ci = metadata["test_ci_95"]

    model_rows = "\n".join(
        f"| {name}{' **(selected)**' if name == best else ''} | "
        f"{r['roc_auc']['mean']:.3f} ± {r['roc_auc']['std']:.3f} | {r['brier_score']['mean']:.3f} |"
        for name, r in sorted(cv.items(), key=lambda kv: -kv[1]["roc_auc"]["mean"])
    )
    fs_rows = "\n".join(
        f"| {name} | {len(metadata['feature_sets'][name])} | "
        f"{r['roc_auc']['mean']:.3f} ± {r['roc_auc']['std']:.3f} |"
        for name, r in metadata["feature_set_comparison"].items()
    )
    cal_rows = "\n".join(
        f"| {name} | {r['brier_score']['mean']:.4f} | {r['roc_auc']['mean']:.3f} |"
        for name, r in metadata["calibration_comparison"].items()
    )
    sub_rows = "\n".join(
        f"| {name} | {g['n']} | {g['positive_rate']:.2f} | "
        f"{'n/a' if g['roc_auc'] is None else f'{g["roc_auc"]:.3f}'} | {g['recall']:.2f} | "
        f"{g['precision']:.2f} |"
        for name, g in metadata["subgroups_oof"].items()
    )

    card = f"""# Model card: heart disease risk model v{metadata["version"]}

_Generated by `python -m heart.train`. Do not edit by hand._

## Intended use
Educational / research demonstration of a screening-style risk estimate for coronary artery
disease (>50% diameter narrowing) from 13 routine clinical measurements.
**Not a medical device.** Not validated for clinical decision-making.

## Model
- **Family:** {best} (Optuna, {metadata["tuning"]["n_trials"]} TPE trials per family)
- **Hyperparameters:** `{metadata["tuning"]["results"][best]["params"]}`
- **Feature set:** {metadata["feature_set"]} ({len(metadata["feature_sets"][metadata["feature_set"]])} features)
- **Probability calibration:** {metadata["calibration"]}
- **Decision threshold:** {metadata["threshold"]:.3f}, the highest threshold that reaches
  {metadata["target_recall"]:.0%} out-of-fold recall (screening: a missed case costs more than a false alarm)

## Data
UCI Heart Disease, Cleveland subset: {metadata["data"]["n_rows"]} patients,
{metadata["data"]["positive_rate"]:.0%} positive. Stratified split: {metadata["data"]["n_train"]} train /
{metadata["data"]["n_test"]} held-out test. Missing `ca`/`thal` (6 values) are imputed inside the pipeline.

## Model selection ({metadata["cv"]["scheme"]}, train set only)
| Model | ROC-AUC | Brier |
|---|---|---|
{model_rows}

### Feature set comparison ({best})
| Feature set | # features | ROC-AUC |
|---|---|---|
{fs_rows}

### Calibration comparison ({best})
| Calibration | Brier (lower is better) | ROC-AUC |
|---|---|---|
{cal_rows}

## Held-out test performance (threshold {metadata["threshold"]:.3f})
| Metric | Value | 95% bootstrap CI |
|---|---|---|
| ROC-AUC | {test["roc_auc"]:.3f} | {ci["roc_auc"][0]:.3f} to {ci["roc_auc"][1]:.3f} |
| Recall (sensitivity) | {test["recall"]:.3f} | {ci["recall"][0]:.3f} to {ci["recall"][1]:.3f} |
| Specificity | {test["specificity"]:.3f} | {ci["specificity"][0]:.3f} to {ci["specificity"][1]:.3f} |
| Precision | {test["precision"]:.3f} | {ci["precision"][0]:.3f} to {ci["precision"][1]:.3f} |
| Brier score | {test["brier_score"]:.3f} | |

At the default 0.5 threshold: recall {metadata["test_metrics_default_threshold"]["recall"]:.3f},
precision {metadata["test_metrics_default_threshold"]["precision"]:.3f}.

## Subgroup performance (out-of-fold on the train set, threshold {metadata["threshold"]:.3f})
| Group | n | Positive rate | ROC-AUC | Recall | Precision |
|---|---|---|---|---|---|
{sub_rows}

## Limitations
- **Small data:** 303 patients from one US hospital (Cleveland Clinic, 1988). The test-set CIs are wide.
- **Population shift:** the cohort is 68% male, mostly middle-aged, and was referred for angiography,
  so it is not a general population. Performance on other populations is unknown.
- **Subgroups:** female patients are under-represented, so their metrics are less reliable.
- **Selection bias:** hyperparameters were tuned on the same CV splits used for model comparison.
  Nested CV would give a less optimistic estimate.
- **Label:** angiographic narrowing, not clinical outcomes.
"""
    path.write_text(card, encoding="utf-8")
    return path
