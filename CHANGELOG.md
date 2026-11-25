# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-11-25

First production-ready release.

### Added
- Installable `heart` package with a single sklearn pipeline (imputation, scaling, one-hot encoding
  and the model) saved as one artifact.
- Training run (`python -m heart.train`):
  - Optuna tuning of 5 model families;
  - repeated stratified CV;
  - CV-based choice of feature set and calibration;
  - a recall-targeted decision threshold;
  - bootstrap confidence intervals and subgroup metrics;
  - generated model card and plots;
  - MLflow tracking;
  - a ROC-AUC quality gate.
- SHAP explanations on the original clinical features, in probability units.
- FastAPI service with `/predict`, `/predict/batch`, `/model-info` and `/health`, request IDs,
  JSON logs and an optional API key.
- Streamlit app with a risk gauge, SHAP waterfall, what-if sliders, PDF report, batch CSV screening
  and a model performance tab.
- Prediction audit log and PSI-based data-drift report (`python -m heart.monitoring`).
- Single Docker image that runs the API or the UI (`SERVICE`, `PORT`), plus docker compose.
- CI: lint, type-check, tests with coverage, retrain behind the quality gate, and an image smoke test.
  Weekly drift workflow.
- Pinned dependency lockfiles (uv), pre-commit hooks and MIT license.

### Fixed
- The app sent category codes the model was never trained on (`cp` 0–3, `thal` 1–3). Inputs now use
  the original UCI codes and invalid codes are rejected.
- SHAP explanations no longer fall back to random values.
- Missing values are imputed after the train/test split, which removes data leakage.

### Removed
- Notebook-only training code with hard-coded paths, the separately pickled scaler and model,
  derived CSVs, and hand-typed comparison charts.

## [0.1.0] - 2025-08-01

### Added
- Initial Streamlit app with an SVM model and SHAP plots.
