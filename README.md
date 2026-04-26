# Environmental Telemetry Risk Pipeline

Small ML pipeline for environmental telemetry data with a clearer structure, dataset validation, model comparison, anomaly detection, and generated reports.

## What The Project Does

- validates the raw dataset before any model step
- cleans and normalizes telemetry-style environmental data
- creates engineered features for risk scoring
- trains and compares baseline classifiers for `risk_level`
- runs anomaly detection with `IsolationForest`
- saves metrics, predictions, artifacts, and plots

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
- `docs/`: architecture and operations notes

## Expected Dataset Columns

The raw CSV is expected to contain:

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

## CLI Usage

Validate the dataset:

```bash
python -m src validate --input data/raw/raw_dataset.csv
```

Train the baseline model:

```bash
python -m src train --input data/raw/raw_dataset.csv
```

Run anomaly detection:

```bash
python -m src anomaly --input data/raw/raw_dataset.csv --contamination 0.05
```

Run the tests:

```bash
python -m pytest
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Operations](docs/OPERATIONS.md)

## Generated Outputs

Training creates:

- `data/outputs/metrics.json`
- `data/outputs/model_comparison.json`
- `data/outputs/predictions.csv`
- `data/outputs/training_summary.md`
- `models/<selected_model>_model.pkl`

Anomaly detection creates:

- `data/outputs/anomalies.csv`
- `data/outputs/anomaly_summary.json`
- `models/anomaly_detector.pkl`
- `reports/figures/anomaly_scatter.png`

## Next Useful Improvements

- richer per-class evaluation
- config via environment variables or YAML
- cross-validation and hyperparameter search
- linting and formatting automation
