import pytest

from heart.config import FEATURES, TARGET
from heart.data import DataValidationError, split_features_target, validate


def test_raw_data_loads_with_expected_shape(raw_df):
    assert raw_df.shape == (303, len(FEATURES) + 1)
    assert set(raw_df[TARGET].unique()) == {0, 1}


def test_question_marks_become_nan_not_imputed(raw_df):
    # Imputation must happen inside the pipeline (after the split), not at load time.
    assert raw_df[["ca", "thal"]].isna().sum().sum() == 6


def test_uses_original_uci_codes(raw_df):
    assert set(raw_df["cp"].unique()) == {1, 2, 3, 4}
    assert set(raw_df["thal"].dropna().unique()) == {3, 6, 7}


def test_validate_rejects_bad_category(raw_df):
    bad = raw_df.copy()
    bad.loc[0, "cp"] = 0  # the old app's encoding
    with pytest.raises(DataValidationError, match="cp"):
        validate(bad)


def test_validate_rejects_out_of_range(raw_df):
    bad = raw_df.copy()
    bad.loc[0, "age"] = 250
    with pytest.raises(DataValidationError, match="age"):
        validate(bad)


def test_split_features_target(raw_df):
    X, y = split_features_target(raw_df)
    assert list(X.columns) == FEATURES
    assert len(X) == len(y)
