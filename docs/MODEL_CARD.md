# Model Card

## Model Overview

This project trains a supervised classifier to predict environmental telemetry `risk_level` and also provides a separate unsupervised anomaly detection pipeline.

The supervised training pipeline compares:

- Logistic Regression
- Random Forest Classifier

The selected supervised model is chosen by mean weighted F1 from stratified cross-validation on the training split only. The winner is then fitted on the full training split and evaluated once on the held-out test split. Test scores do not influence model selection.

Cross-validation uses up to three folds, limited by the smallest training class. If any training class has fewer than two observations, training stops with a clear error rather than selecting a model using test scores. Ties keep the first declared candidate for reproducibility.

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

Generated training artifacts include:

- `data/outputs/metrics.json`
- `data/outputs/model_comparison.json`
- `data/outputs/model_metadata.json`
- `data/outputs/predictions.csv`
- `data/outputs/batch_predictions.csv`
- `data/outputs/training_summary.md`
- `models/<selected_model>_model.pkl`

## Evaluation

The pipeline reports:

- accuracy
- weighted precision
- weighted recall
- weighted F1
- confusion matrix
- per-class classification report
- compact stratified cross-validation summary

Mean cross-validation weighted F1 is used for model selection because it combines precision and recall while accounting for class support. `model_comparison.json` and `candidate_models` in `metrics.json` contain training cross-validation statistics with explicit `_mean` and `_std` keys. Headline and per-class metrics in `metrics.json` describe only the selected model on the held-out test split. Metadata records the selection metric, the datasets used for selection and evaluation, raw `feature_columns`, `engineered_feature_columns`, and preprocessing steps. The serialized artifact is a fitted `Pipeline` (`telemetry_pipeline_v1`), with its own input schema and learned preprocessing statistics.

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
- Random stratified splitting does not measure future-time generalization; chronological evaluation requires a separate design and enough historical data.

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
