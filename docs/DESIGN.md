# Design notes

Why the system is built the way it is. For the model's numbers and limitations, see the
[model card](../models/MODEL_CARD.md). It is regenerated on every training run.

## Data and validation

- **Original UCI codes end to end.** `cp` is 1–4, `thal` is 3/6/7, and so on. The codes are defined
  once in `heart/config.py`, and the training pipeline, the Pydantic input schema and the UI all read
  them from there. An earlier version of the app sent codes the model had never seen (`cp` 0–3,
  `thal` 1–3). The schema now rejects those, and a test pins that behaviour.
- **Missing values are imputed inside the pipeline.** Imputation statistics come only from the
  training fold, so nothing leaks from validation or test data.
- **One artifact.** Preprocessing, the model, the decision threshold and the SHAP background sample
  are saved together in `heart_model.joblib`. Training and serving cannot disagree on preprocessing.

## Model selection

- **Five families are tuned with Optuna (TPE)**, then compared on repeated stratified 5×3 CV using
  different splits from the tuning runs. The held-out test set is scored once, after every decision
  has been made.
- **Logistic regression was selected.** All tuned families land within about one standard deviation
  of each other (ROC-AUC 0.891–0.898). With 303 patients, extra capacity buys nothing, and the linear
  model is better calibrated and easier to explain. The CV ranking was accepted as it came out.
- **Feature set by CV, not by eye.** The original EDA dropped 5 features after looking at plots.
  On CV, all 13 features scored 0.898 and the 8 picked by eye scored 0.894, so all 13 are kept.
- **Calibration only when it helps.** Sigmoid calibration was compared on CV Brier score and kept only
  if lower. It was not lower (0.1282 vs 0.1275), so the model is uncalibrated.
  SVM probabilities always come from `CalibratedClassifierCV`.

## Decision threshold

The use case is screening. A missed case costs far more than a follow-up test. The threshold is
therefore the **highest one that reaches 90% recall on out-of-fold predictions** of the training set,
which gives the best precision at that recall. It is rounded *down* so the recall target still holds
after rounding. A test caught the rounding-up version. The risk bands are tied to this threshold, so
"Low" always means screen-negative.

## Explanations

Permutation SHAP runs on the 13 original inputs, not the one-hot columns. It works in probability
units, against 100 real training patients. That makes outputs like "Major vessels = 3: +24
percentage points" readable by a clinician. The values add up exactly to the prediction
(`base + Σ SHAP = p`), and a unit test checks this. One explainer is shared across threads, so calls
are serialised with a lock.

## Serving

- **One inference module** (`heart.predict`) is used by the API, the UI and the batch scorer.
- **One container image**, with the service chosen by `SERVICE=api|ui` at runtime and `PORT`
  honoured. That makes it work on any container host without per-service build targets.
- **The API** validates requests with Pydantic (422 with field-level errors), caps batch size,
  supports an optional `X-API-Key`, tags every request with an `X-Request-ID`, and logs requests as
  JSON lines.
- **Prediction audit log.** Clinical inputs and outputs are appended as JSON lines, with no
  identifiers. This log is what drift monitoring reads.

## Quality gates

- Training aborts, and saves nothing, if the best CV ROC-AUC is below 0.85.
- CI retrains from scratch on every push and re-runs the model-quality tests.
- A test fails if the committed model was pickled with a different scikit-learn version than the one
  installed. The lockfile keeps the two in sync.

## Monitoring

`heart.monitoring` computes the Population Stability Index for each feature, comparing logged
inputs with the training data. The cut-offs are under 0.10 stable, 0.10–0.25 moderate and over 0.25
drift. A scheduled workflow runs it weekly with `--fail-on-drift`.

## Known limitations

- The data is small, old and from a single site: 303 angiography patients, Cleveland, 1988.
- The test set has 61 patients, so its confidence intervals are wide.
- Recall is lower for women (0.78 vs 0.94 out-of-fold).
- Tuning and model comparison share CV data. Nested CV would be stricter.
- What-if sliders show how the model responds, not causal treatment effects.
