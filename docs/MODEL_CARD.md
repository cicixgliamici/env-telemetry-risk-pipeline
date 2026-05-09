# Model Card

## Model Overview

This project trains a supervised classifier to predict environmental telemetry `risk_level` and also provides a separate unsupervised anomaly detection pipeline.

The supervised training pipeline compares:

- Logistic Regression
- Random Forest Classifier

The selected supervised model is chosen by weighted F1 score on the held-out test split. Cross-validation summaries are generated to give reviewers a more stable view than a single split alone.

## Intended Use

The model is intended as a reviewable baseline for environmental telemetry risk classification. It is suitable for:

- demonstrating ML pipeline structure
- validating schema and preprocessing assumptions
- comparing simple supervised baselines
- producing local training artifacts
- supporting portfolio or reviewer discussion

It is not intended as a production safety system without real domain data, calibration, monitoring, and expert validation.

## Inputs

Expected raw columns:

- `timestamp`
- `humidity_pct`
- `temperature_c`
- `rainfall_mm`
- `pollution_proxy`
- `salinity_proxy`
- `pressure_hpa`
- `vibration_mm_s`
- `exposure_hours`
- `risk_level`
- `failure_risk`

The model uses engineered telemetry features derived from the raw numeric inputs. The columns `risk_level`, `failure_risk`, and `timestamp` are excluded from supervised model features.

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

Weighted F1 is used for model selection because it balances precision and recall while accounting for class support.

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
