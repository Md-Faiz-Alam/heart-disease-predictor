import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from heart.data import load_raw, split_features_target
from heart.features import build_pipeline
from heart.predict import HeartModel
from heart.train import make_bundle


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    return load_raw()


@pytest.fixture(scope="session")
def trained_model(raw_df) -> HeartModel:
    X, y = split_features_target(raw_df)
    pipeline = build_pipeline(LogisticRegression(max_iter=2000)).fit(X, y)
    return HeartModel(make_bundle(pipeline, X, "logistic_regression", "test"))


@pytest.fixture
def patient_values() -> dict:
    return dict(
        age=67,
        sex=1,
        cp=4,
        trestbps=160,
        chol=286,
        fbs=0,
        restecg=2,
        thalach=108,
        exang=1,
        oldpeak=1.5,
        slope=2,
        ca=3,
        thal=3,
    )
