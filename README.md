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

Model selection uses mean weighted F1 from cross-validation on the training split only. The default protocol uses random stratified holdout and stratified folds; `--split-strategy temporal` uses the latest timestamps as holdout and expanding historical training folds. The selected model is fitted on the full training split, then evaluated once on the held-out test split. `model_comparison.json` contains candidate cross-validation means and standard deviations; final test metrics belong only to the winner. Stratified CV requires at least two observations per training class; temporal CV requires at least two classes in every historical training window.

Each supervised model is saved as a complete scikit-learn `Pipeline`: median imputation, feature engineering, optional standard scaling for logistic regression, and classification. Imputation and scaling are fitted separately inside each training fold and frozen during prediction. The fitted artifact contains its raw feature schema, so new pipeline predictions do not require `model_metadata.json`.

Training compares five configurations on identical folds: both classifier families with raw or engineered features, plus a majority-class `DummyClassifier`. Reports include weighted and macro F1, class-1 risk recall, feature-ablation deltas, and the selected model's improvement over the dummy. A [reproducible 1,000-row synthetic experiment](docs/EXPERIMENT_RESULTS.md) records the measured results and their limits.

## Project Layout

- `src/`: package code
- `tests/`: pytest suite
- `data/raw/`: source CSV data
- `runs/<run_id>/`: configuration, environment, processed data, models, metrics, and figures
- `data/outputs/`: default standalone profiling and batch prediction outputs
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
python -m src predict --input data/raw/raw_dataset.csv --model "runs/<run_id>/models/<selected_model>_model.pkl"
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

Replace the prediction example's placeholders with the `model_artifact` path printed
by training. Training and anomaly detection print their `run_id` and absolute `run_dir`.

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

Evaluate on future timestamps using a larger synthetic review dataset:

```bash
python scripts/generate_synthetic_data.py --output data/raw/synthetic_dataset.csv --rows 1000 --seed 42
python -m src train --input data/raw/synthetic_dataset.csv --split-strategy temporal --test-size 0.2 --temporal-gap 1
```

Temporal evaluation reserves the latest fraction of distinct timestamps, keeps
simultaneous observations together, and uses the same expanding historical folds
for every candidate. `--temporal-gap` excludes timestamp groups before each CV
validation window and before the final test window. The bundled 12-row example
has insufficient class diversity in early history for this protocol. See
[Temporal Evaluation](docs/TEMPORAL_EVALUATION.md) for boundaries, recorded evidence,
and limitations. Synthetic results remain implementation checks.

Create a data profile:

```bash
python -m src profile --input data/raw/raw_dataset.csv
```

Run batch prediction after training:

```bash
python -m src predict --input data/raw/raw_dataset.csv --model "runs/<run_id>/models/<selected_model>_model.pkl"
```

`predicted_risk_probability` is the model probability for class `1` (`RISK_LABEL`),
located using the saved model's `classes_` order, including for multiclass models.
The column is omitted when class `1`, probability prediction, or class ordering is
unavailable; another class is never substituted. These probabilities are not calibrated.

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
- [Temporal Evaluation](docs/TEMPORAL_EVALUATION.md): chronological holdout, expanding folds, timestamp groups, and exclusion gaps

## Review Notes

This is intentionally a small, auditable baseline rather than a highly tuned model package. The code emphasizes explicit validation, reproducible defaults, simple model comparison, security-aware checks, deployment smoke tests, and clear generated artifacts. The included dataset is small, so model metrics should be read as pipeline smoke evidence rather than production-grade performance claims. Use the synthetic data generator when a larger deterministic dataset is useful for review.

## Generated Outputs

Training and anomaly detection each create a unique folder under `runs/` relative
to the working directory. Use `--output-dir` to change the parent directory:

```bash
python -m src train --input data/raw/raw_dataset.csv --output-dir runs
python -m src anomaly --input data/raw/raw_dataset.csv --output-dir runs
```

Every run contains `run.json` (identity, timestamps, and status), `config.json`,
`environment.json` (Python, platform, installed package versions), and
`requirements.txt` with exact installed package versions. This is an environment
snapshot, including unrelated installed packages, rather than a portable lockfile.
Reproduction requires the same Python/platform, source revision, and input data;
install this repository separately because its local package is excluded from the pins.

Inside each training run:

- `processed/cleaned_dataset.csv` and `processed/featured_dataset.csv`
- `outputs/metrics.json`, `outputs/model_comparison.json`, and `outputs/model_metadata.json`
- `outputs/evaluation_protocol.json`: holdout and CV row membership, class counts, time ranges, and gaps
- `outputs/predictions.csv` and `outputs/training_summary.md`
- `models/<selected_model>_model.pkl`
- `figures/confusion_matrix.png` and, for tree models, `figures/feature_importance.png`

Inside each anomaly run:

- `outputs/anomalies.csv` and `outputs/anomaly_summary.json`
- `models/anomaly_detector.pkl`
- `figures/anomaly_scatter.png`

Runs never reuse an existing directory. Failed runs retain partial artifacts with
status `failed`; interrupted processes may leave status `running`. Existing artifacts
in the old fixed directories are retained, but new training and anomaly runs use the
run folders. Profiling and batch prediction remain separate commands with explicit
output options; their default outputs stay in `data/outputs/`.

## Next Useful Improvements

- experiments on sourced observational data with defensible risk labels
- config via environment variables or YAML
- hyperparameter search within training-only cross-validation
- linting and formatting automation
