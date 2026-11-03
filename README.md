# Environmental Telemetry Risk Pipeline

Small ML pipeline for environmental telemetry data with a clearer structure, dataset validation, model comparison, anomaly detection, and generated reports.

## What The Project Does

- validates the raw dataset before any model step
- cleans and normalizes telemetry-style environmental data
- creates engineered features for risk scoring
- trains and compares baseline classifiers for `risk_level`
- profiles data quality and target balance
- runs batch predictions from a saved model
- runs anomaly detection with `IsolationForest`
- saves metrics, predictions, artifacts, and plots

Model selection uses mean weighted F1 from stratified cross-validation on the training split only. The selected model is fitted on the full training split, then evaluated once on the held-out test split. `model_comparison.json` contains candidate cross-validation means and standard deviations; final test metrics belong only to the winner. Training requires at least two observations per class in the training split.

Each supervised model is saved as a complete scikit-learn `Pipeline`: median imputation, feature engineering, optional standard scaling for logistic regression, and classification. Imputation and scaling are fitted separately inside each training fold and frozen during prediction. The fitted artifact contains its raw feature schema, so new pipeline predictions do not require `model_metadata.json`.

Training compares five configurations on identical folds: both classifier families with raw or engineered features, plus a majority-class `DummyClassifier`. Reports include weighted and macro F1, class-1 risk recall, feature-ablation deltas, and the selected model's improvement over the dummy. A [reproducible 1,000-row synthetic experiment](docs/EXPERIMENT_RESULTS.md) records the measured results and their limits.

## Project Layout

- `src/`: package code
- `tests/`: pytest suite
- `data/raw/`: source CSV data
- `data/processed/`: cleaned and enriched datasets
- `data/outputs/`: metrics, predictions, and anomaly outputs
- `models/`: serialized models
- `reports/figures/`: generated charts
- `notebooks/`: exploratory notebooks
- `scripts/`: lightweight helper entrypoints
- `Dockerfile`: minimal runtime image for deployment smoke tests
- `docs/`: architecture and operations notes

## Expected Dataset Columns

Every CSV must contain these telemetry columns, including for prediction and anomaly detection:

- `timestamp`
- `humidity_pct`
- `temperature_c`
- `rainfall_mm`
- `pollution_proxy`
- `salinity_proxy`
- `pressure_hpa`
- `vibration_mm_s`
- `exposure_hours`

Training additionally requires a complete `risk_level` target with at least two classes. `failure_risk` is always optional and excluded from model inputs. Prediction, anomaly detection, and profiling accept unlabeled data and batches containing just one row or one observed class.

Missing telemetry values are allowed for imputation. Infinite values, missing timestamps, and measurements outside the broad physical ranges are rejected. See the [Dataset Card](docs/DATASET_CARD.md) for exact bounds and label rules.

## Setup

```bash
pip install -e .[dev]
```

If you prefer a requirements file:

```bash
pip install -r requirements.txt
```

On Windows PowerShell, a clean setup usually looks like this:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .[dev]
```

If PowerShell blocks activation, you can use:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

## Reviewer Quickstart

The fastest way to verify the project is:

```bash
pip install -e .[dev]
python -m src validate --input data/raw/raw_dataset.csv
python -m pytest -q
```

For a fuller smoke run:

```bash
python scripts/generate_synthetic_data.py --output data/raw/synthetic_dataset.csv --rows 250 --seed 42
python -m src validate --input data/raw/synthetic_dataset.csv
python -m src profile --input data/raw/synthetic_dataset.csv
python -m src train --input data/raw/raw_dataset.csv
python -m src predict --input data/raw/raw_dataset.csv --model models/random_forest_model.pkl
python -m src anomaly --input data/raw/raw_dataset.csv --contamination 0.05
```

Expected high-level behavior:

- validation prints a JSON status with row and column counts
- profiling writes data quality JSON and Markdown reports
- training writes cleaned data, engineered data, metrics, predictions, plots, and a model file
- batch prediction writes a CSV with predicted risk labels
- anomaly detection writes row-level anomaly labels, a summary JSON, a plot, and a detector file
- synthetic data generation creates a reproducible larger review dataset when the bundled sample is too small
- generated runtime artifacts are ignored by git

## CLI Usage

Validate the dataset:

```bash
python -m src validate --input data/raw/raw_dataset.csv
```

Validate new telemetry without requiring labels:

```bash
python -m src validate --input data/raw/incoming.csv --purpose prediction
```

`--purpose` accepts `training` (the default), `prediction`, or `anomaly`. The `train`, `predict`, and `anomaly` commands select their own validation rules automatically; profiling uses telemetry-only checks.

Train the baseline model:

```bash
python -m src train --input data/raw/raw_dataset.csv
```

Create a data profile:

```bash
python -m src profile --input data/raw/raw_dataset.csv
```

Run batch prediction after training:

```bash
python -m src predict --input data/raw/raw_dataset.csv --model models/random_forest_model.pkl
```

Run anomaly detection:

```bash
python -m src anomaly --input data/raw/raw_dataset.csv --contamination 0.05
```

Run the tests:

```bash
python -m pytest
```

Generate a reproducible synthetic dataset:

```bash
python scripts/generate_synthetic_data.py --output data/raw/synthetic_dataset.csv --rows 250 --seed 42
```

Build and smoke-test the runtime container:

```bash
docker build -t env-telemetry-risk-pipeline:local .
docker run --rm env-telemetry-risk-pipeline:local
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Operations](docs/OPERATIONS.md)
- [Modeling Notes](docs/MODELING_NOTES.md): mathematical background, diagrams, metric definitions, leakage rationale, anomaly detection notes, and common reviewer questions
- [Model Card](docs/MODEL_CARD.md): intended use, evaluation, limitations, and monitoring notes
- [Dataset Card](docs/DATASET_CARD.md): schema, synthetic data, validation rules, and leakage notes
- [Reviewer Checklist](docs/REVIEWER_CHECKLIST.md): commands and review signals to inspect
- [Experiment Results](docs/EXPERIMENT_RESULTS.md): fixed protocol, dummy baseline, feature ablation, results, and reproduction identity

## Review Notes

This is intentionally a small, auditable baseline rather than a highly tuned model package. The code emphasizes explicit validation, reproducible defaults, simple model comparison, security-aware checks, deployment smoke tests, and clear generated artifacts. The included dataset is small, so model metrics should be read as pipeline smoke evidence rather than production-grade performance claims. Use the synthetic data generator when a larger deterministic dataset is useful for review.

## Generated Outputs

Training creates:

- `data/outputs/metrics.json`
- `data/outputs/model_comparison.json`
- `data/outputs/model_metadata.json`
- `data/outputs/data_profile.json`
- `data/outputs/data_profile.md`
- `data/outputs/predictions.csv`
- `data/outputs/batch_predictions.csv`
- `data/outputs/training_summary.md`
- `models/<selected_model>_model.pkl`

Anomaly detection creates:

- `data/outputs/anomalies.csv`
- `data/outputs/anomaly_summary.json`
- `models/anomaly_detector.pkl`
- `reports/figures/anomaly_scatter.png`

## Next Useful Improvements

- experiments on sourced observational data with defensible risk labels
- config via environment variables or YAML
- hyperparameter search within training-only cross-validation
- linting and formatting automation
