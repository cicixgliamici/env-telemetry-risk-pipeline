# Architecture

## Overview

The project is organized as a lightweight Python package with a simple CLI entrypoint. The goal is to keep the runtime path easy to follow while still separating responsibilities clearly enough for maintenance and extension.

## Execution Flow

1. `src.cli` parses the user command.
2. `src.data_loader` loads the raw CSV and delegates schema checks to `src.validation`, using the operation's purpose. Training requires a complete target; prediction, anomaly detection, and profiling require only telemetry. All paths reject invalid ranges, infinities, missing timestamps, and ambiguous normalized headers.
3. For supervised training, `src.preprocessing` normalizes, deduplicates, and sorts rows without imputing missing inputs, then creates the train/test split.
4. `src.modeling` builds candidate pipelines that impute inputs, add derived features, optionally scale, and classify. Cross-validation fits each pipeline on its fold's training observations only.
5. `src.train` selects a candidate using training-only cross-validation. Optional cost-based threshold selection uses the same training folds before the winner is refitted and evaluated on the holdout. The independent anomaly path retains its existing batch cleaning and feature engineering.
6. `src.evaluate`, `src.visualize`, and `src.utils` write metrics and artifacts.

## Module Responsibilities

- `src/config.py`: project paths and core constants
- `src/artifacts.py`: exclusive run directories, lifecycle manifest, configuration, and environment snapshot
- `src/validation.py`: file-level and schema-level input validation
- `src/data_loader.py`: CSV loading after validation
- `src/preprocessing.py`: cleaning and train/test split preparation
- `src/splitting.py`: stratified or temporal partitions, expanding windows, and auditable evaluation membership
- `src/feature_engineering.py`: derived telemetry features
- `src/modeling.py`: supervised pipelines with fitted imputation and optional standard scaling
- `src/decision.py`: optional persisted binary decision cutoff and training-validation cost selection
- `src/decision_evaluation.py`: decision artifacts and original-versus-selected held-out evaluation
- `src/predict.py`: batch inference using the saved pipeline and its embedded input schema
- `src/train.py`: supervised model comparison and artifact generation
- `src/training_artifacts.py`: split data container, fitted-feature audit export, metadata, model persistence, and final reports
- `src/synthetic_data.py`: packaged deterministic telemetry generator, also exposed by the helper script
- `src/review.py`: reproducible runtime bundle and relative report links across all review experiments
- `src/anomaly.py`: unsupervised anomaly detection pipeline
- `src/robustness_reference.py`: frozen source model, original holdout membership, and consistency checks
- `src/stress_scenarios.py`: predetermined measurement/case-mix scenarios with training-only scales
- `src/robustness.py`: repeated stress evaluation, metric deltas, row audit, and reviewer report
- `src/evaluate.py`: model metrics and comparable summaries
- `src/experiment.py`: data fingerprint and versions, paired feature deltas, and reviewer-facing training report
- `src/visualize.py`: chart generation
- `src/utils.py`: filesystem, logging, and timestamp helpers
- `src/cli.py`: operator-facing commands

## Production Notes

- Validation fails early before model execution.
- Training and anomaly artifacts are written to unique folders under the configurable run root.
- Each run keeps its configuration, environment snapshot, and lifecycle status next to its artifacts.
- Supervised `.pkl` artifacts contain the entire fitted pipeline. New artifacts do not require an external metadata file for prediction; older bare classifiers retain a compatibility path with their original preprocessing behavior.
- Prediction preserves input row order and duplicates and never fits the new pipeline on scoring data.
- Logging is intentionally simple and console-first.
- Current configuration is code-based; environment-driven config would be the next step for deployment flexibility.
