"""Preprocessing: one ColumnTransformer saved together with the model."""

from collections.abc import Sequence

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from heart.config import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    CATEGORY_LABELS,
    FEATURES,
    NUMERIC_FEATURES,
)


def build_preprocessor(features: Sequence[str] = FEATURES) -> ColumnTransformer:
    """Columns not in `features` are dropped, so any feature subset can be evaluated."""
    numeric_cols = [f for f in NUMERIC_FEATURES if f in features]
    binary_cols = [f for f in BINARY_FEATURES if f in features]
    categorical_cols = [f for f in CATEGORICAL_FEATURES if f in features]

    transformers = []
    if numeric_cols:
        numeric = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]
        )
        transformers.append(("num", numeric, numeric_cols))
    if binary_cols:
        transformers.append(("bin", SimpleImputer(strategy="most_frequent"), binary_cols))
    if categorical_cols:
        categorical = Pipeline(
            [
                ("impute", SimpleImputer(strategy="most_frequent")),
                (
                    "onehot",
                    OneHotEncoder(
                        # Fixed category lists: an unseen code at predict time is a bug.
                        categories=[
                            [float(code) for code in sorted(CATEGORY_LABELS[c])]
                            for c in categorical_cols
                        ],
                        handle_unknown="error",
                        sparse_output=False,
                    ),
                ),
            ]
        )
        transformers.append(("cat", categorical, categorical_cols))

    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def build_pipeline(model: BaseEstimator, features: Sequence[str] = FEATURES) -> Pipeline:
    return Pipeline([("preprocess", build_preprocessor(features)), ("model", model)])
