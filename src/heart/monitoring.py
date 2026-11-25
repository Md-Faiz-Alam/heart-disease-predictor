"""Data drift check: compare logged production inputs with the training distribution.

Uses the Population Stability Index (PSI) per feature:
    PSI < 0.10  stable · 0.10-0.25  moderate shift · > 0.25  significant shift

Usage:
    python -m heart.monitoring                       # reads HEART_PREDICTION_LOG_PATH
    python -m heart.monitoring --fail-on-drift       # exit 1 on significant drift (for CI/cron)
"""

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from heart.config import CATEGORY_LABELS, FEATURE_LABELS, FEATURES, NUMERIC_FEATURES
from heart.data import load_raw, split_features_target
from heart.settings import get_settings

PSI_WARN = 0.10
PSI_DRIFT = 0.25
MIN_SAMPLES = 30
_EPS = 1e-4


@dataclass(frozen=True)
class FeatureDrift:
    feature: str
    label: str
    psi: float
    status: str
    reference_mean: float
    current_mean: float


def _status(psi: float) -> str:
    if psi >= PSI_DRIFT:
        return "drift"
    if psi >= PSI_WARN:
        return "warning"
    return "ok"


def _psi(ref_share: np.ndarray, cur_share: np.ndarray) -> float:
    ref = np.clip(ref_share, _EPS, None)
    cur = np.clip(cur_share, _EPS, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def psi(reference: pd.Series, current: pd.Series, categorical: bool, bins: int = 10) -> float:
    reference, current = reference.dropna(), current.dropna()
    if categorical:
        categories = sorted(set(reference) | set(current))
        ref_share = reference.value_counts(normalize=True).reindex(categories, fill_value=0)
        cur_share = current.value_counts(normalize=True).reindex(categories, fill_value=0)
        return _psi(ref_share.to_numpy(), cur_share.to_numpy())

    # Quantile bins from the reference; open-ended outer bins catch out-of-range values.
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts = np.histogram(reference, edges)[0] / len(reference)
    cur_counts = np.histogram(current, edges)[0] / len(current)
    return _psi(ref_counts, cur_counts)


def drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> list[FeatureDrift]:
    results = []
    for feature in FEATURES:
        categorical = feature in CATEGORY_LABELS and feature not in NUMERIC_FEATURES
        value = psi(reference[feature], current[feature], categorical)
        results.append(
            FeatureDrift(
                feature=feature,
                label=FEATURE_LABELS[feature],
                psi=round(value, 4),
                status=_status(value),
                reference_mean=round(float(reference[feature].mean()), 3),
                current_mean=round(float(current[feature].mean()), 3),
            )
        )
    return sorted(results, key=lambda r: -r.psi)


def load_logged_inputs(path: Path) -> pd.DataFrame:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line)["inputs"])
    return pd.DataFrame(rows, columns=FEATURES).astype(float)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check production inputs for data drift")
    parser.add_argument("--log", type=Path, default=get_settings().prediction_log_path)
    parser.add_argument("--output", type=Path, help="write the report as JSON")
    parser.add_argument("--fail-on-drift", action="store_true")
    args = parser.parse_args(argv)

    if not args.log.exists():
        print(f"No prediction log at {args.log}")
        return 0
    current = load_logged_inputs(args.log)
    if len(current) < MIN_SAMPLES:
        print(f"Only {len(current)} logged predictions; need at least {MIN_SAMPLES}.")
        return 0

    reference, _ = split_features_target(load_raw())
    report = drift_report(reference, current)

    print(f"Drift report: {len(current)} logged predictions vs {len(reference)} training rows\n")
    print(f"{'feature':<40} {'PSI':>7}  {'status':<8} {'train mean':>10} {'live mean':>10}")
    for r in report:
        print(
            f"{r.label:<40} {r.psi:>7.3f}  {r.status:<8} {r.reference_mean:>10.2f} "
            f"{r.current_mean:>10.2f}"
        )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {"n_current": len(current), "features": [asdict(r) for r in report]}, indent=2
            ),
            encoding="utf-8",
        )

    drifted = [r.feature for r in report if r.status == "drift"]
    if drifted:
        print(f"\nSignificant drift in: {', '.join(drifted)}")
        if args.fail_on_drift:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
