"""Load and validate the UCI Cleveland heart disease dataset."""

from pathlib import Path

import pandas as pd

from heart.config import (
    CATEGORY_LABELS,
    FEATURES,
    NUMERIC_RANGES,
    RAW_COLUMNS,
    RAW_DATA_PATH,
    TARGET,
)


class DataValidationError(ValueError):
    """Raised when a dataset does not match the expected schema."""


def load_raw(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Read the raw CSV (no header, '?' for missing) and binarise the target.

    Missing values are left as NaN on purpose: imputation happens inside the model
    pipeline, after the train/test split, so no test information leaks into training.
    """
    df = pd.read_csv(path, header=None, names=RAW_COLUMNS, na_values="?")
    df[TARGET] = (df[TARGET] > 0).astype(int)
    validate(df)
    return df


def validate(df: pd.DataFrame) -> None:
    """Check columns, value ranges, category codes and target values."""
    errors: list[str] = []

    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(f"Missing columns: {missing}")

    for col, (low, high) in NUMERIC_RANGES.items():
        values = df[col].dropna()
        bad = values[~values.between(low, high)]
        if not bad.empty:
            errors.append(f"{col}: {len(bad)} value(s) outside [{low}, {high}], e.g. {bad.iloc[0]}")

    for col, labels in CATEGORY_LABELS.items():
        values = df[col].dropna()
        bad = values[~values.isin(list(labels))]
        if not bad.empty:
            errors.append(f"{col}: unexpected code(s) {sorted(bad.unique().tolist())}")

    if not df[TARGET].isin([0, 1]).all():
        errors.append(f"{TARGET}: must be 0/1")

    if errors:
        raise DataValidationError("Dataset failed validation:\n  " + "\n  ".join(errors))


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[FEATURES].astype(float), df[TARGET]
