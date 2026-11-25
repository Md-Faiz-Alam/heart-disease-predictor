"""Single source of truth for paths, feature definitions and clinical encodings.

All codes below are the ORIGINAL UCI Cleveland encodings. The model, the input schema
and the UI all read from here, so they cannot drift apart.
"""

import os
from pathlib import Path

# Project root. Defaults to the repo checkout; set HEART_HOME when the package is installed
# into site-packages (e.g. in Docker), where __file__ no longer points at the repo.
ROOT = Path(os.environ.get("HEART_HOME", Path(__file__).resolve().parents[2]))
RAW_DATA_PATH = ROOT / "data" / "raw" / "uci_heart_disease.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "heart_model.joblib"
METADATA_PATH = MODEL_DIR / "metadata.json"
MODEL_CARD_PATH = MODEL_DIR / "MODEL_CARD.md"
REPORTS_DIR = ROOT / "reports"
MLFLOW_DIR = ROOT / "mlruns"
MLFLOW_EXPERIMENT = "heart-disease"

RANDOM_STATE = 42
TEST_SIZE = 0.2
SHAP_BACKGROUND_SIZE = 100

# Model selection / tuning
N_TRIALS = 40  # Optuna trials per model family
CV_FOLDS = 5
CV_REPEATS = 3
DEFAULT_THRESHOLD = 0.5
# Screening use case: missing a sick patient is worse than a false alarm. The decision
# threshold is the highest one whose out-of-fold recall still reaches this target.
TARGET_RECALL = 0.90
# Quality gate: training fails (and nothing is saved) below this cross-validated ROC-AUC.
MIN_CV_ROC_AUC = 0.85

TARGET = "target"
FEATURES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]
RAW_COLUMNS = [*FEATURES, TARGET]

# Candidate feature sets compared by cross-validation. "eda_selected" is the subset the
# original notebook picked by eye; it has to beat "all" on CV to be used.
FEATURE_SETS: dict[str, list[str]] = {
    "all": FEATURES,
    "eda_selected": ["age", "sex", "cp", "thalach", "exang", "oldpeak", "ca", "thal"],
}

# Continuous / ordinal -> scaled. `ca` (0-3 vessels) is ordinal, so it is kept numeric.
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
BINARY_FEATURES = ["sex", "fbs", "exang"]
# Nominal codes -> one-hot. Scaling these would invent an order that does not exist.
CATEGORICAL_FEATURES = ["cp", "restecg", "slope", "thal"]

# Plausible clinical input ranges (inclusive). Used for data validation and the input schema.
NUMERIC_RANGES: dict[str, tuple[float, float]] = {
    "age": (18, 100),
    "trestbps": (80, 220),
    "chol": (100, 600),
    "thalach": (60, 220),
    "oldpeak": (0.0, 7.0),
    "ca": (0, 3),
}

CATEGORY_LABELS: dict[str, dict[int, str]] = {
    "sex": {0: "Female", 1: "Male"},
    "cp": {
        1: "Typical angina",
        2: "Atypical angina",
        3: "Non-anginal pain",
        4: "Asymptomatic",
    },
    "fbs": {0: "No (<= 120 mg/dl)", 1: "Yes (> 120 mg/dl)"},
    "restecg": {
        0: "Normal",
        1: "ST-T wave abnormality",
        2: "Left ventricular hypertrophy",
    },
    "exang": {0: "No", 1: "Yes"},
    "slope": {1: "Upsloping", 2: "Flat", 3: "Downsloping"},
    "thal": {3: "Normal", 6: "Fixed defect", 7: "Reversible defect"},
}

FEATURE_LABELS: dict[str, str] = {
    "age": "Age (years)",
    "sex": "Sex",
    "cp": "Chest pain type",
    "trestbps": "Resting blood pressure (mm Hg)",
    "chol": "Serum cholesterol (mg/dl)",
    "fbs": "Fasting blood sugar > 120 mg/dl",
    "restecg": "Resting ECG",
    "thalach": "Max heart rate achieved (bpm)",
    "exang": "Exercise-induced angina",
    "oldpeak": "ST depression (exercise vs rest)",
    "slope": "Slope of peak exercise ST segment",
    "ca": "Major vessels coloured by fluoroscopy",
    "thal": "Thallium stress test",
}
