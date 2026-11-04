# Model Card

## Model Overview

This project trains a supervised classifier to predict environmental telemetry `risk_level` and also provides a separate unsupervised anomaly detection pipeline.

The supervised training pipeline compares:

- Logistic Regression with raw and engineered features
- Random Forest Classifier with raw and engineered features
- a majority-class `DummyClassifier` reference

The selected supervised model is chosen by mean weighted F1 from cross-validation on the training split only. The default uses stratified random splitting and stratified CV. The optional temporal strategy uses the latest timestamps as holdout and expanding historical CV windows, keeping equal timestamps together. The winner is then fitted on the full training split and evaluated once on the held-out test split. Test scores do not influence model selection.

Stratified cross-validation uses up to three identical folds for all candidates, limited by the smallest training class. If any training class has fewer than two observations, training stops with a clear error rather than selecting a model using test scores. Temporal CV uses up to three expanding windows and requires at least two classes in every historical training window. An optional timestamp-group gap applies to both CV and holdout. Ties keep the first declared candidate for reproducibility. Unsuffixed family names retain engineered features; `_raw` variants bypass feature engineering. The dummy is a selectable candidate rather than a model the other candidates are assumed to beat.

## Intended Use

The model is intended as a reviewable baseline for environmental telemetry risk classification. It is suitable for:

- demonstrating ML pipeline structure
- validating schema and preprocessing assumptions
- comparing simple supervised baselines
- producing local training artifacts
- supporting portfolio or reviewer discussion

It is not intended as a production safety system without real domain data, calibration, monitoring, and expert validation.

## Inputs

Telemetry columns required by the CSV interface:

- `timestamp`
- `humidity_pct`
- `temperature_c`
- `rainfall_mm`
- `pollution_proxy`
- `salinity_proxy`
- `pressure_hpa`
- `vibration_mm_s`
- `exposure_hours`

Training additionally requires a non-missing `risk_level` target with at least two non-negative integer classes. `failure_risk` is optional. Prediction and anomaly detection do not require either label and accept single-row or single-class batches. Optional label annotations may be missing; any supplied values must be valid classes. See the Dataset Card for numeric bounds and timestamp requirements.

The saved pipeline accepts raw numeric telemetry inputs. It imputes missing inputs before creating engineered features. The columns `risk_level`, `failure_risk`, and `timestamp` are excluded from supervised model features.

Each cross-validation fold learns its own median imputations. Logistic regression also learns standard scaling after feature engineering; Random Forest uses unscaled features. The winning pipeline is refitted on the full training split. Prediction reuses these fitted statistics without recomputing them on the incoming batch. A feature missing throughout a training fold retains its column and is filled with zero, a reproducible fallback rather than a domain estimate.

## Output

The supervised model predicts `risk_level`.

Batch predictions include `predicted_risk_probability` only when the model supports
`predict_proba`, exposes `classes_`, and includes `RISK_LABEL = 1`. The probability
column is selected by class label, including for multiclass and single-class models.
If class 1 or its column mapping is unavailable, the probability column is omitted
rather than assigning another class's probability. These probabilities are not calibrated.

Generated training artifacts include:

- `runs/<run_id>/outputs/metrics.json`
- `runs/<run_id>/outputs/model_comparison.json`
- `runs/<run_id>/outputs/model_metadata.json`
- `runs/<run_id>/outputs/predictions.csv`
- `runs/<run_id>/outputs/training_summary.md`
- `runs/<run_id>/models/<selected_model>_model.pkl`

The run root is configurable with `--output-dir`. Each run also records its effective
configuration, Python/platform, installed package versions, and completion status.
Batch prediction is a separate command and defaults to `data/outputs/batch_predictions.csv`.

## Evaluation

The pipeline reports:

- accuracy
- weighted precision
- weighted recall
- weighted F1
- macro F1, giving equal weight to each observed class
- class-1 risk recall and its test support; `null`/`N/A` when no class-1 observations exist
- confusion matrix
- per-class classification report
- stratified or chronological cross-validation summary and exact evaluation protocol
- raw-versus-engineered feature deltas on identical training folds
- selected mean CV weighted F1 minus the dummy baseline

`RISK_LABEL = 1` defines the bundled task's risk class. Other numeric class codes are not automatically reinterpreted as elevated risk. Confusion matrix labels are saved explicitly alongside the matrix. Risk recall and macro F1 provide diagnostic evidence; weighted F1 remains the selection objective, so the winner need not maximize risk recall.

Mean cross-validation weighted F1 is used for model selection because it combines precision and recall while accounting for class support. `model_comparison.json` and `candidate_models` in `metrics.json` contain training cross-validation statistics with explicit `_mean` and `_std` keys. Headline and per-class metrics in `metrics.json` describe only the selected model on the held-out test split. Metadata records the selection metric, the datasets used for selection and evaluation, raw `feature_columns`, `engineered_feature_columns`, and preprocessing steps. The serialized artifact is a fitted `Pipeline` (`telemetry_pipeline_v1`), with its own input schema and learned preprocessing statistics.

Metadata also records `feature_set`, a SHA-256 of the normalized table actually used, class counts, and Python/library versions. These identify the experiment but do not verify the origin of the data. For raw-only or dummy winners, `engineered_feature_columns` contains the actual eight model input columns after imputation, and the `feature_set` explicitly says `raw`. See [Experiment Results](EXPERIMENT_RESULTS.md) for a measured synthetic benchmark and its provenance limitations.

## Ethical And Safety Notes

This model should not be used to make safety-critical environmental or infrastructure decisions without:

- real representative data
- domain expert review
- calibration and threshold analysis
- monitoring for drift
- human review of high-risk predictions
- clear escalation procedures

The synthetic dataset is useful for testing and demonstrations, but it does not prove real-world model performance.

## Limitations

- The bundled raw dataset is very small.
- Feature thresholds are heuristic.
- The model is not calibrated for operational probability decisions.
- No production data drift monitor is active yet.
- The model artifact is local and does not yet use a registry.
- Cross-validation improves review confidence but does not replace representative data.
- Zero-filling a feature absent throughout a training fold preserves the schema but does not establish that zero is a physically meaningful value.
- Random stratified splitting does not measure future-time generalization. Optional temporal evaluation requires enough historical class diversity and does not isolate sites or account automatically for delayed labels and overlapping label horizons; see [Temporal Evaluation](TEMPORAL_EVALUATION.md).

## Recommended Monitoring

If this model were deployed, monitor:

- input schema failures
- feature distribution drift
- prediction distribution drift
- anomaly rate over time
- false positives and false negatives after human review
- model version and feature list used for each prediction

## Maintenance Notes

Before promoting a new model version:

- run the full test suite
- run lint and security-oriented static checks
- regenerate training artifacts
- compare cross-validation metrics
- inspect per-class metrics
- verify feature columns in `model_metadata.json`
- document any material data or model changes
